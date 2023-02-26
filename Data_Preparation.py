
#%%
import os
import pickle
import numpy as np

# Define the path to the dataset directory
dataset_dir = "/home/bakary/Desktop/My_GitHub/Defensive_distillation/Defensive-Distillation-as-an-Adversarial-Roustness-Algorithm/traffic-signs-data/"

# Load the training set
with open(os.path.join(dataset_dir, "train.p"), "rb") as f:
    train_data = pickle.load(f)

x_train, y_train = train_data["features"], train_data["labels"]



# Load the validation set
with open(os.path.join(dataset_dir, "valid.p"), "rb") as f:
    valid_data = pickle.load(f)

x_val, y_val = valid_data["features"], valid_data["labels"]



# Load the testing set
with open(os.path.join(dataset_dir, "test.p"), "rb") as f:
    test_data = pickle.load(f)

x_test, y_test = test_data["features"], test_data["labels"]



x_test.shape, y_test.shape, x_train.shape, y_train.shape
#%%
