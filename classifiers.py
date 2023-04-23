# Classifiers

# This notebook contains the implementation of our three deep neural classifiers with correct hyperparameters.

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd


train_features = pd.read_csv('train_features.csv', delimiter=',') # load the features after creating them
test_feautres = pd.read_csv('test_features.csv', delimiter=',') # load the features after creating them

train_labels = pd.read_csv("../data/training-set.csv")["is_suicide"]
test_labels = pd.read_csv("../data/testing-set.csv")["is_suicide"]

# training hyperparameters

epochs = 80
batch_size = 32

# Convolutional Neural Network

cnn_path = "cnn"
filters = 3
kernel_size = 2

cnn_model = tf.keras.Sequential([
    layers.Input(shape=(512,768)),
    layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(64, activation='relu', kernel_initializer='he_uniform'),
    layers.Dense(1, activation='sigmoid')
])

cnn_model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

cnn_model.summary()

mc = ModelCheckpoint(cnn_path + ".h5", monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

# Fully Dense Network

dense_path = "dense"

dense_model = tf.keras.Sequential([
    layers.Input(shape=(512,)),
    layers.Dense(128, activation='relu', kernel_initializer='he_uniform'),
    layers.Dense(64, activation='relu', kernel_initializer='he_uniform'),
    layers.Dense(1, activation='sigmoid')
])

dense_model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

dense_model.summary()

mc = ModelCheckpoint(dense_path + ".h5", monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

# Bi-LSTM

bilstm_path = "bilstm"
pool_size = 2

bilstm_model = tf.keras.Sequential([
    layers.Input(shape=(512,768)),
    layers.Bidirectional(layers.LSTM(20, return_sequences=True, dropout=0.25, recurrent_dropout=0.2)),
    layers.MaxPooling1D(pool_size=pool_size),
    layers.Flatten(),
    layers.Dense(10, activation='relu', kernel_initializer='he_uniform'),
    layers.Dense(1, activation='sigmoid')
])

bilstm_model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

bilstm_model.summary()

mc = ModelCheckpoint(bilstm_path + ".h5", monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)


bilstm.summary()
