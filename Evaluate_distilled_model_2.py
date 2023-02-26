## This file will evaluate the distilled model (student model) 
# with ensemble adversarial attacks

import os
import pickle
import numpy as np
import tensorflow as tf

# Define the path to the adversarial examples directory
adv_examples_dir = "/home/bakary/Desktop/My_GitHub/Defensive_distillation/Defensive-Distillation-as-an-Adversarial-Roustness-Algorithm/traffic-signs-data/"

# Load the adversarial examples
with open(os.path.join(adv_examples_dir, "adversarial.p"), "rb") as f:
    adversarial_data = pickle.load(f)

x_adv, y_adv = adversarial_data["features"], adversarial_data["labels"]

# Load the student model
student_model = tf.keras.models.load_model("/home/bakary/Desktop/My_GitHub/Defensive_distillation/Defensive-Distillation-as-an-Adversarial-Roustness-Algorithm/model.h5")

# Evaluate the student model on the adversarial examples
adv_loss, adv_acc = student_model.evaluate(x_adv, y_adv)

print("Adversarial loss:", adv_loss)
print("Adversarial accuracy:", adv_acc)