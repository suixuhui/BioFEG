# BioFEG

## Data Preparation
* Cleate a `data` fold
* Download the datasets from <https://github.com/dhdhagar/arboEL>, place it under `data`
* Prepare data: `python data_process.py`

## Run
### Biencoder
We train our biencoder in an iterative way: train biencoder -> train gan -> generate latent features -> finetune biencoder. We train our biencoder on in-batch negatives in the first iteration and on hard negatives in the following iterations.
1. train biencoder

train biencoder on in-batch negatives: `PYTHONPATH=. python blink/biencoder/train_biencoder.py --data_path data/medmentions/processed --output_path models/medmentions/biencoder --learning_rate 1e-05 --num_train_epochs 1 --train_batch_size 64 --eval_batch_size 32 --bert_model SapBERT-from-PubMedBERT-fulltext --type_optimization all_encoder_layers`

train biencoder on hard negatives: `PYTHONPATH=. python blink/biencoder/train_biencoder_hard.py --data_path models/medmentions/finetune/top64_candidates --output_path models/medmentions/biencoder --path_to_model models/medmentions/finetune/pytorch_model.bin --learning_rate 1e-05 --num_train_epochs 1 --train_batch_size 4 --eval_batch_size 4 --bert_model SapBERT-from-PubMedBERT-fulltext --type_optimization all_encoder_layers`

2. train gan `PYTHONPATH=. python blink/biencoder/train_gan.py --data_path data/medmentions/processed --output_path models/medmentions/gan --path_to_model models/medmentions/biencoder/pytorch_model.bin --learning_rate 5e-05 --num_train_epochs 80 --train_batch_size 64 --eval_batch_size 128 --bert_model SapBERT-from-PubMedBERT-fulltext --type_optimization all_encoder_layers`

3. generate latent features

generate latent features on in-batch negatives: `PYTHONPATH=. python blink/biencoder/generate_features.py --data_path data/medmentions/processed --entity_dict_path data/medmentions/documents/all_documents.json --gan_path models/gan/epoch_49 --output_path models/medmentions/gan --path_to_model models/medmentions/biencoder/pytorch_model.bin --encode_batch_size 64 --top_k 64 --bert_model SapBERT-from-PubMedBERT-fulltext`

generate latent features on hard negatives: `PYTHONPATH=. python blink/biencoder/generate_features_hard.py --data_path data/medmentions/processed --entity_dict_path data/medmentions/documents/all_documents.json --gan_path models/gan/epoch_49 --output_path models/medmentions/gan --path_to_model models/medmentions/biencoder/pytorch_model.bin --encode_batch_size 64 --top_k 64 --bert_model SapBERT-from-PubMedBERT-fulltext`

4. finetune biencoder

finetune biencoder on in-batch negatives: `PYTHONPATH=. python blink/biencoder/finetune.py --data_path data/medmentions/processed --generate_data_path models/medmentions/gan/candidates_50/ --output_path models/medmentions/finetune --path_to_model models/medmentions/finetune/pytorch_model.bin --learning_rate 1e-09 --num_train_epochs 1 --train_batch_size 64 --eval_batch_size 32 --bert_model SapBERT-from-PubMedBERT-fulltext --type_optimization all_encoder_layers`

finetune biencoder on hard negatives: `PYTHONPATH=. python blink/biencoder/finetune_hard.py --data_path data/medmentions/processed --generate_data_path models/medmentions/gan/candidates_50/ --output_path models/medmentions/finetune --path_to_model models/medmentions/finetune/pytorch_model.bin --learning_rate 1e-09 --num_train_epochs 1 --train_batch_size 64 --eval_batch_size 32 --bert_model SapBERT-from-PubMedBERT-fulltext --type_optimization all_encoder_layers`

eval biencoder: We eval biencoder after training biencoder or finetuning biencoder. `PYTHONPATH=. python blink/biencoder/eval_biencoder.py --data_path data/medmentions/processed --entity_dict_path data/medmentions/documents/all_documents.json --output_path models/medmentions --path_to_model models/medmentions/finetune/pytorch_model.bin --encode_batch_size 64 --eval_batch_size 8 --top_k 64 --bert_model SapBERT-from-PubMedBERT-fulltext --mode train,valid,test,test_seen,test_unseen`

### Cross-encoder


If you use our code in your work, please cite us.

*Xuhui Sui, Ying Zhang, Xiangrui Cai, Kehui Song, Baohang Zhou, Xiaojie Yuan and Wensheng Zhang. BioFEG: Generate Latent Features for Biomedical Entity Linking. Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP 2023).*
