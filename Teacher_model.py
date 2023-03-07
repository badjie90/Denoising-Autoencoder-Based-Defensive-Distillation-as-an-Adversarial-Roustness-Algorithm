#%%
from tensorflow import keras
from keras import layers
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input, Softmax
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
#from keras.datasets import gtsrb
from Data_Preparation import *
from Distillation_model import defensive_distillation
from Denoising_autoencoder import x_train_clean
import tensorflow as tf
import numpy as np


# The Teacher model



teacher_model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(43)
])


X, y = defensive_distillation(x_train_clean, y_train)


teacher_model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

teacher_model.fit(X, y, epochs=20)

 









