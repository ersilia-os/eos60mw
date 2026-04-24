import sys
import types
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# dgl 2.1.0's graphbolt module uses torchdata subpackages removed in torchdata >= 0.8.0.
# Pre-register dgl.graphbolt as an empty stub so dgl skips loading it entirely.
# graphbolt is only needed for scalable training pipelines, not for inference.
if "dgl.graphbolt" not in sys.modules:
    sys.modules["dgl.graphbolt"] = types.ModuleType("dgl.graphbolt")

import os
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pickle
import numpy as np
import dill
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
from transformers import RobertaTokenizerFast
import deepchem as dc
from deepchem.data import NumpyDataset
from ersilia_pack_utils.core import read_smiles, write_out

input_file = sys.argv[1]
output_file = sys.argv[2]

root = os.path.dirname(os.path.abspath(__file__))
checkpoints = os.path.abspath(os.path.join(root, "..", "..", "checkpoints"))
_TOKENIZER_PATH = os.path.join(checkpoints, "smiles-tokenizer")

# --- Read SMILES ---
_, smiles_list = read_smiles(input_file)

# --- Validate SMILES upfront ---
valid_smiles = []
valid_indices = []
for i, smi in enumerate(smiles_list):
    if Chem.MolFromSmiles(smi) is not None:
        valid_smiles.append(smi)
        valid_indices.append(i)

# --- Featurization ---

def rdkit_fps(smiles_list):
    return np.array(
        [np.array(AllChem.RDKFingerprint(Chem.MolFromSmiles(smi), fpSize=2048), dtype=np.float32)
         for smi in smiles_list]
    )


# --- Sklearn models (RF, MLP, GB) ---

def sklearn_scores(model_path, fps):
    with open(model_path, "rb") as f:
        clf = pickle.load(f)
    return clf.predict_proba(fps)[:, 1].tolist()


# --- GCN model ---

def gcn_scores(smiles_list, model_path):
    with open(model_path, "rb") as f:
        model = dill.load(f)
    model.model_dir_is_temp = False  # prevents __del__ from trying to clean up a non-existent tmp dir
    featurizer = dc.feat.MolGraphConvFeaturizer()
    features = featurizer.featurize(smiles_list)
    dataset = NumpyDataset(X=features, y=None, ids=None)
    proba = model.predict(dataset)  # shape: (N, 2)
    return proba[:, 1].tolist()


# --- ChemBERTa models ---

def chemberta_scores(smiles_list, model_path):
    tokenizer = RobertaTokenizerFast.from_pretrained(_TOKENIZER_PATH)
    with open(model_path, "rb") as f:
        model = dill.load(f)
    model.eval()
    scores = []
    with torch.no_grad():
        for smi in smiles_list:
            try:
                inputs = tokenizer(smi, truncation=True, padding=True, return_tensors="pt")
                out = model(**inputs)
                prob_active = F.softmax(out.logits, dim=1).squeeze()[1].item()
            except Exception:
                prob_active = np.nan
            scores.append(prob_active)
    return scores


# --- Initialize output array with NaN ---
n = len(smiles_list)
outputs = np.full((n, 6), np.nan, dtype=np.float32)

# --- Run all 6 endpoints on valid SMILES only ---
if valid_smiles:
    fps = rdkit_fps(valid_smiles)

    nan_col = [np.nan] * len(valid_smiles)

    try:
        rf_s = sklearn_scores(os.path.join(checkpoints, "Leishmania_RF.pkl"), fps)
    except Exception:
        rf_s = nan_col[:]
    try:
        mlp_s = sklearn_scores(os.path.join(checkpoints, "Leishmania_MLP.pkl"), fps)
    except Exception:
        mlp_s = nan_col[:]
    try:
        gb_s = sklearn_scores(os.path.join(checkpoints, "Coronavirus_GB.pkl"), fps)
    except Exception:
        gb_s = nan_col[:]
    try:
        gcn_s = gcn_scores(valid_smiles, os.path.join(checkpoints, "Coronavirus_GCN.pkl"))
    except Exception:
        gcn_s = nan_col[:]
    try:
        cl_s = chemberta_scores(valid_smiles, os.path.join(checkpoints, "Leishmania_ChemBERTa.pkl"))
    except Exception:
        cl_s = nan_col[:]
    try:
        cc_s = chemberta_scores(valid_smiles, os.path.join(checkpoints, "Coronavirus_ChemBERTa.pkl"))
    except Exception:
        cc_s = nan_col[:]

    for j, i in enumerate(valid_indices):
        outputs[i] = [rf_s[j], mlp_s[j], cl_s[j], gcn_s[j], gb_s[j], cc_s[j]]

# --- Write output ---
header = [
    "leishmania_rf",
    "leishmania_mlp",
    "leishmania_chemberta",
    "coronavirus_gcn",
    "coronavirus_gb",
    "coronavirus_chemberta",
]

write_out(outputs, header, output_file, np.float32)
