"""tfexample: A Flower / TensorFlow app."""

import os

import keras
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
import pandas as pd
import numpy as np
from transformers import BertTokenizer

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# vocab_size = 5000
# max_length = 100
# oov_token = "<OOV>"
# embedding_dim = 64

# Parameters
# MAX_WORDS = 5000  # Vocabulary size
MAX_LEN = 100  # Max sequence length
EMBEDDING_DIM = 128  # Embedding dimension

def load_model(learning_rate: float = 0.01):

    model = keras.Sequential([
        layers.Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_LEN),
        layers.LSTM(64, return_sequences=False, dropout=0.2),
        layers.Dense(1, activation="sigmoid")
    ])
    
    # model = Sequential([
    #     Embedding(vocab_size, embedding_dim, input_length=max_length),
    #     Conv1D(128, 5, activation='relu'),
    #     GlobalMaxPooling1D(),
    #     Dropout(0.3),
    #     Dense(64, activation='relu'),
    #     Dropout(0.3),
    #     Dense(1, activation='sigmoid')  # Binary classification
    # ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        # loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Ensure the model is built before setting weights
    model.build(input_shape=(None, MAX_LEN))

    return model


# Load dataset
def load_full_dataset():
    df = pd.read_csv("processed_combined.csv")  # Ensure the file is in the working directory
    texts = df["text"].astype(str).tolist()  # Assuming the column name is 'text'
    labels = df["label"].astype(int)   # Assuming 'label' is the target column

   # Load Pretrained BERT Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    tokens = tokenizer(texts, padding="max_length", truncation=True, max_length=100, return_tensors="tf")
    
    sequences = tokens['input_ids']

    return sequences, labels, tokenizer

# Tokenize and preprocess dataset
X, y, tokenizer = load_full_dataset()
vocab_size = tokenizer.vocab_size 

# Manually partition dataset into clients
# def partition_data(X, y, num_partitions):
#     indices = np.arange(len(X))
#     np.random.shuffle(indices)

#     partitions = np.array_split(indices, num_partitions)
#     partitioned_data = [(X[idx], y[idx]) for idx in partitions]
    
#     return partitioned_data

def partition_data(X, y, num_partitions):
    """
    Partitions the dataset (X, y) into num_partitions.
    """
    X, y = np.array(X), np.array(y)  # Convert to NumPy arrays (if not already)

    partitions = np.array_split(np.arange(len(X)), num_partitions)  # Create partitions

    # Corrected indexing
    partitioned_data = [(X[idx.tolist()], y[idx.tolist()]) for idx in partitions]

    return partitioned_data

# fds = None  # Cache FederatedDataset


# def load_data(partition_id, num_partitions):
#     # Download and partition dataset
#     # Only initialize `FederatedDataset` once
#     global fds
#     if fds is None:
#         partitioner = IidPartitioner(num_partitions=num_partitions)
#         fds = FederatedDataset(
#             dataset={"X": X, "y": y},  # Using in-memory dataset
#             partitioners={"train": partitioner},
#         )

#     partition = fds.load_partition(partition_id, "train")
#     partition.set_format("numpy")

#     # Divide data on each node: 90% train, 10% test
#     partition = partition.train_test_split(test_size=0.1)
#     x_train, y_train = partition["train"]["X"], partition["train"]["y"]
#     x_test, y_test = partition["test"]["X"], partition["test"]["y"]

#     return x_train, y_train, x_test, y_test
def load_data(partition_id, num_partitions):
    # Create partitions
    client_partitions = partition_data(X, y, num_partitions)

    x_train, y_train = client_partitions[partition_id]
    
    # Split into 90% training, 10% test
    split = int(0.9 * len(x_train))
    x_test, y_test = x_train[split:], y_train[split:]
    x_train, y_train = x_train[:split], y_train[:split]

    return x_train, y_train, x_test, y_test