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
from Data_Preparation import *
from Student_model import student_model
from keras.losses import SparseCategoricalCrossentropy
import tensorflow as tf
import numpy as np
#%%
# Define the path to the dataset directory
dataset_dir = "/home/bakary/Desktop/My_GitHub/Defensive_distillation/Defensive-Distillation-as-an-Adversarial-Roustness-Algorithm/traffic-signs-data/"



# Load the testing set
with open(os.path.join(dataset_dir, "test.p"), "rb") as f:
    test_data = pickle.load(f)

x_test, y_test = test_data["features"], test_data["labels"]



# Define the epsilon values for the attacks
epsilons = [0.01, 0.02, 0.03]



# Define the FGSM attack function
def fgsm_attack(model, x, y, epsilon):
    """
    Fast Gradient Sign Method attack
    """
    x_adv = tf.Variable(x)
    with tf.GradientTape() as tape:
        tape.watch(x_adv)
        logits = model(x_adv)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, logits)
    gradients = tape.gradient(loss, x_adv)
    perturbation = tf.sign(gradients)
    x_adv = x_adv + epsilon*perturbation
    x_adv = tf.clip_by_value(x_adv, 0, 1)
    return x_adv.numpy()



# Define the I-FGSM attack function
def ifgsm_attack(model, x, y, epsilon, alpha=0.01, num_iter=10):
    """
    Iterative Fast Gradient Sign Method attack
    """
    x_adv = tf.Variable(x)
    for i in range(num_iter):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            logits = model(x_adv)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y, logits)
        gradients = tape.gradient(loss, x_adv)
        perturbation = alpha*tf.sign(gradients)
        x_adv = x_adv + perturbation
        x_adv = tf.clip_by_value(x_adv, x-epsilon, x+epsilon)
        x_adv = tf.clip_by_value(x_adv, 0, 1)
    return x_adv.numpy()



# Define the function to generate adversarial examples using an ensemble of attacks
def generate_ensemble_adversarial(model, x, y, epsilons):
    """
    Generate adversarial examples using an ensemble of FGSM and I-FGSM attacks
    """
    x_adv = x.copy()
    for epsilon in epsilons:
        x_adv_fgsm = fgsm_attack(model, x_adv, y, epsilon)
        x_adv_ifgsm = ifgsm_attack(model, x_adv, y, epsilon)
        x_adv = (x_adv + x_adv_fgsm + x_adv_ifgsm) / 3.0
    return x_adv



# Generate adversarial examples using the ensemble of attacks
x_adv = generate_ensemble_adversarial(student_model, x_test, y_test, epsilons)

# Save the adversarial examples to your directory
with open(os.path.join(dataset_dir, "adversarial.p"), "wb") as f:
    pickle.dump({"features": x_adv, "labels": y_test}, f)

