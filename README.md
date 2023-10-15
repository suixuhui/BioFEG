# BioFEG

## Data Preparation
* Cleate a `data` fold
* Download the datasets from <https://github.com/dhdhagar/arboEL>, place it under `data`
* Prepare data: `python data_process.py`

## Run
### Biencoder
We train our biencoder in an iterative way: train biencoder -> train gan -> generate latent features -> finetune biencoder. We train our biencoder on in-batch negatives in the first iteration and on hard negatives in the following iterations.
1. train biencoder

train biencoder on in-batch negatives: `PYTHONPATH=. python blink/biencoder/train_biencoder.py --data_path data/medmentions/processed --output_path models/medmentions/biencoder --learning_rate 1e-05 --num_train_epochs 1 --max_context_length 128 --max_cand_length 128 --train_batch_size 64 --eval_batch_size 32 --bert_model SapBERT-from-PubMedBERT-fulltext --type_optimization all_encoder_layers`

train biencoder on hard negatives: `PYTHONPATH=. python blink/biencoder/train_biencoder_hard.py --data_path models/medmentions/finetune/top64_candidates --output_path models/medmentions/hard_1/biencoder --path_to_model models/medmentions/finetune/pytorch_model.bin --learning_rate 1e-05 --num_train_epochs 1 --max_context_length 128 --max_cand_length 128 --train_batch_size 4 --eval_batch_size 4 --bert_model SapBERT-from-PubMedBERT-fulltext --type_optimization all_encoder_layers`

2. train gan 
### Cross-encoder


If you use our code in your work, please cite us.

*Xuhui Sui, Ying Zhang, Xiangrui Cai, Kehui Song, Baohang Zhou, Xiaojie Yuan and Wensheng Zhang. BioFEG: Generate Latent Features for Biomedical Entity Linking. Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP 2023).*
