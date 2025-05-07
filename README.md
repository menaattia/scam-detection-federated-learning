# Telephone Scam Detection Using a Privacy-Preserving Federated Learning Approach

## Datasets

1. The Phone Scam dataset on Hugging Face was created by running the code in `scripts/synthetic_data_script.ipynb`. The dataset is in the `data/phone-scam-dataset` folder.

2. Telephone-Based Scam Signature Dataset found in the `data/telephone-scam-data` folder.

## Data Preprocessing

The `scripts/data_preprocessing.ipynb`script was used to preprocess both datasets. Processed datasets are in the `data/processed_data` folder.

## Centralized Models

The following files in the `centralized_models` folder were used to train and evaluate XGBOOST, LSTM, and CNN models.
`cnn_model.ipynb`
`lstm_model.ipynb`
`xgboost_model.ipynb`

- Note: the Jupyter notebooks were run on Google Colab. Running them locally may require installing packages.

## Federated Learning

Code adapted from Flower tutorials [here](https://github.com/adap/flower/tree/main/examples/quickstart-tensorflow)

To run the code: run `flwr run` inside the `federated_learning_cnn` and `federated_learning_lstm` folders.

Datasets: `data\federated_data` combines the train and test sets of each of the two datasets used as well as both of them combined.
