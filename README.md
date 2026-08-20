# CidalsDB anti-Leishmania and anti-SARS-CoV/SARS-CoV-2 activity model

Scores compounds against Leishmania parasites and SARS-CoV-2 across six models drawn from cidalsDB, a curated resource of anti-pathogen bioactivity. Harigua-Souiai and colleagues assembled published screening data into a consistent database and fitted classifiers on top, with the explicit aim of supporting research on neglected diseases where screening capacity is scarce. Leishmaniasis and coronavirus data differ greatly in volume, so confidence is not uniform across the six outputs.

This model was incorporated on 2026-03-25.Last packaged on 2026-04-24.

## Information
### Identifiers
- **Ersilia Identifier:** `eos60mw`
- **Slug:** `cidalsdb`

### Domain
- **Task:** `Annotation`
- **Subtask:** `Activity prediction`
- **Biomedical Area:** `COVID-19`, `Leishmaniasis`
- **Target Organism:** `Leishmania major`, `SARS-CoV-2`
- **Tags:** `Antimicrobial activity`, `Antiviral activity`, `Sars-CoV-2`

### Input
- **Input:** `Compound`
- **Input Dimension:** `1`

### Output
- **Output Dimension:** `6`
- **Output Consistency:** `Fixed`
- **Interpretation:** Probabilities of activity against Leishmania species and SARS-CoV-2 across six curated models.

Below are the **Output Columns** of the model:
| Name | Type | Direction | Description |
|------|------|-----------|-------------|
| leishmania_rf | float | high | Random Forest probability of anti-Leishmania activity |
| leishmania_mlp | float | high | MLP probability of anti-Leishmania activity |
| leishmania_chemberta | float | high | ChemBERTa probability of anti-Leishmania activity |
| coronavirus_gcn | float | high | GCN probability of anti-coronavirus activity |
| coronavirus_gb | float | high | Gradient Boosting probability of anti-coronavirus activity |
| coronavirus_chemberta | float | high | ChemBERTa probability of anti-coronavirus activity |


### Source and Deployment
- **Source:** `Local`
- **Source Type:** `External`
- **DockerHub**: [https://hub.docker.com/r/ersiliaos/eos60mw](https://hub.docker.com/r/ersiliaos/eos60mw)
- **Docker Architecture:** `AMD64`
- **S3 Storage**: [https://ersilia-models-zipped.s3.eu-central-1.amazonaws.com/eos60mw.zip](https://ersilia-models-zipped.s3.eu-central-1.amazonaws.com/eos60mw.zip)

### Resource Consumption
- **Model Size (Mb):** `661`
- **Environment Size (Mb):** `5798`
- **Image Size (Mb):** `7758.22`

**Computational Performance (seconds):**
- 10 inputs: `33.2`
- 100 inputs: `31.93`
- 10000 inputs: `712.03`

### References
- **Source Code**: [https://github.com/Harigua/CidalsDB/tree/main](https://github.com/Harigua/CidalsDB/tree/main)
- **Publication**: [https://doi.org/10.1186/s13321-024-00929-7](https://doi.org/10.1186/s13321-024-00929-7)
- **Publication Type:** `Peer reviewed`
- **Publication Year:** `2024`
- **Ersilia Contributor:** [arnaucoma24](https://github.com/arnaucoma24)

### License
This package is licensed under a [GPL-3.0](https://github.com/ersilia-os/ersilia/blob/master/LICENSE) license. The model contained within this package is licensed under a [GPL-3.0-only](LICENSE) license.

**Notice**: Ersilia grants access to models _as is_, directly from the original authors, please refer to the original code repository and/or publication if you use the model in your research.


## Use
To use this model locally, you need to have the [Ersilia CLI](https://github.com/ersilia-os/ersilia) installed.
The model can be **fetched** using the following command:
```bash
# fetch model from the Ersilia Model Hub
ersilia fetch eos60mw
```
Then, you can **serve**, **run** and **close** the model as follows:
```bash
# serve the model
ersilia serve eos60mw
# generate an example file
ersilia example -n 3 -f my_input.csv
# run the model
ersilia run -i my_input.csv -o my_output.csv
# close the model
ersilia close
```

## About Ersilia
The [Ersilia Open Source Initiative](https://ersilia.io) is a tech non-profit organization fueling sustainable research in the Global South.
Please [cite](https://github.com/ersilia-os/ersilia/blob/master/CITATION.cff) the Ersilia Model Hub if you've found this model to be useful. Always [let us know](https://github.com/ersilia-os/ersilia/issues) if you experience any issues while trying to run it.
If you want to contribute to our mission, consider [donating](https://www.ersilia.io/donate) to Ersilia!
