# Reddit Social Network Dataset Research based on GNN

## Abstract

The Reddit Hyperlink Network dataset analyzes inter-community interactions and conflicts across 36,000 Reddit communities over 40 months, including 1.8 billion user comments. Previous work by Kumar *et al.* \cite{Social-LSTM} introduced a Social-LSTM model to predict community conflicts using graph embeddings, user activity, and textual features. Building on this foundation, our project explores novel embedding techniques using Graph Neural Networks (GCNs) and evaluates multiple downstream classification methods, including Transformers and Social-LSTMs. By systematically comparing initialization methods and loss functions, we aim to enhance the predictive performance for inter-community conflict detection and user mobilization patterns.

## Acknowledgment

Our work is built upon the foundation laid by Kumar *et al.* in the paper:
**Community Interaction and Conflict on the Web**
Srijan Kumar, William L. Hamilton, Jure Leskovec, and Dan Jurafsky.
*Proceedings of the 2018 World Wide Web Conference (WWW '18)*
[DOI: 10.1145/3178876.3186141](https://doi.org/10.1145/3178876.3186141)

This seminal work highlights the mechanisms of community interactions and conflict dynamics on Reddit, providing the basis for dataset analysis and conflict prediction models.

------

## Installation and Setup

### Requirements

To use this project, ensure you have the following setup:

- **CUDA 11.8** for GPU-based training.
- For CPU-based training, use the environment specified in `environment.yml`.

Install the required dependencies:

```
conda env create -f environment.yml
conda activate <environment_name>
```

------

## Dataset Download

Download the Reddit Hyperlink dataset using the following command:

```
python download_and_extract.py
```

------

## Training the Embeddings

To train user-community embeddings using GCN, run the following command:

```
python train_embedding.py --loss_type 2 --embedding_type 2 --negative_sample 5
```

Alternatively, for sigmoid-based loss and pre-trained embeddings:

```
python train_embedding.py --loss_type 3 --embedding_type 3 --negative_sample 5
```

The model logs will be saved to `output.log`.

------

## Downstream Classification Tasks

Train and evaluate the Social-LSTM model using the learned embeddings:

```
python social_lstm_model.py --epochs 20 --learning_rate 0.002 --enable_mean_pooling false \
--model 1 --warmup_steps 200 --dropout 0.2 --final_layer_social --final_dense --enable_scheduler \
--include_text --loss_type 3 --embedding_type 2
```

For a Transformer-based model with tuned hyperparameters:

```
python social_lstm_model.py --epochs 20 --learning_rate 0.05 --enable_mean_pooling false \
--model 7 --warmup_steps 200 --dropout 0.5 --final_layer_social --final_dense --enable_scheduler
```

------

## Best Performance

The overall best performance for embedding training and downstream classification is:

- **Validation AUROC**: 0.733