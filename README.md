# BioFEG

## Data Preparation
* Cleate a `data` fold
* Download the datasets from <https://github.com/dhdhagar/arboEL>, place it under `data`
* Prepare data: `python data_process.py`

## Run
### Biencoder
We train our biencoder in an iterative way: train biencoder -> train gan -> generate latent features -> finetune biencoder. We train our biencoder on in-batch negatives in the first iteration and on hard negatives in the following negatives.
### Cross-encoder


If you use our code in your work, please cite us.

*Xuhui Sui, Ying Zhang, Xiangrui Cai, Kehui Song, Baohang Zhou, Xiaojie Yuan and Wensheng Zhang. BioFEG: Generate Latent Features for Biomedical Entity Linking. Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP 2023).*
