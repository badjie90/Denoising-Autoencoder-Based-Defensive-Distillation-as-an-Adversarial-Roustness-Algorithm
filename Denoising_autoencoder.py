#%%
import os
import pickle
import numpy as np
from tensorflow import keras
from keras import layers
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input, Softmax
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from Data_Preparation import *
#from Teacher_model import *
#from Distillation_model import *
#from keras.datasets import gtsrb
import tensorflow as tf
import numpy as np


# Define the autoencoder

# dae_model = tf.keras.Sequential([
#     layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
#     layers.MaxPooling2D((2, 2), padding='same'),
#     layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
#     layers.MaxPooling2D((2, 2), padding='same'),
#     layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
#     layers.MaxPooling2D((2, 2), padding='same'),
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(128, activation='relu'),
#     layers.Dense(256, activation='relu'),
#     layers.Dense(1024, activation='relu'),
#     layers.Reshape((8, 8, 16)),
#     layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same'),
#     layers.UpSampling2D((2, 2)),
#     layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same'),
#     layers.UpSampling2D((2, 2)),
#     layers.Conv2DTranspose(3, (3, 3), activation='relu', padding='same'),
#     #layers.UpSampling2D((2, 2)),
#     #layers.Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')
# ])





# Define the encoder
encoder = tf.keras.Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(16, activation='relu'),
])

# Define the decoder
decoder = tf.keras.Sequential([
    layers.Input(shape=(16,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(4*4*128, activation='relu'),
    layers.Reshape((4, 4, 128)),
    layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.UpSampling2D((2, 2)),
    layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.UpSampling2D((2, 2)),
    layers.Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same'),
])

# Define the denoising autoencoder as a single model
dae_model = tf.keras.Sequential([encoder, decoder])




# Train the DAE model on the combination of the original train and test datasets:

# Combine the training and test data

#x_all = np.concatenate((x_train, x_val), axis=0)
x_all = np.concatenate((X_train_perturb, x_val), axis=0)


# Train the DAE on the combined data
dae_model.compile(optimizer='adam', loss='mse')
#dae_model.fit(x_all, x_all, epochs=20)
print(x_all.shape)
print(x_val.shape)
print(dae_model(x_all[0:5]).shape)
dae_model.fit(x_all, x_all, epochs=20, validation_data=(x_val, x_val))



# Evaluate the DAE's performance on the test set. You can also use validation set here if you wish.
x_test_clean = dae_model.predict(x_test)
test_loss = np.mean(np.square(x_test - x_test_clean))
print("Validation loss:", test_loss)

                 #OR

# x_val_clean = dae_model.predict(x_val)
# val_loss = np.mean(np.square(x_val - x_val_clean))
# print("Validation loss:", val_loss)


# Filter out poison data samples from the training data
x_train_clean = dae_model.predict(x_train)


# %%
