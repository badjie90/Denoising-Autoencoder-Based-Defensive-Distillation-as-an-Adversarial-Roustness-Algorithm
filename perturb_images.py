#%%
import os
import pickle
import tensorflow as tf
from keras.models import load_model

# Define the path to the dataset directory
# Define the path to the dataset directory
dataset_dir = "/home/lasige/Desktop/MY_GitHub/Ensemble_model_Training/traffic-signs-data"

# Load the training set
with open(os.path.join(dataset_dir, "train.p"), "rb") as f:
    train_data = pickle.load(f)

# Load the pre-trained model
model = load_model("/home/lasige/Desktop/MY_GitHub/Ensemble_model_Training/model2.hdf5")

# Define the attack parameters
epsilon_fgsm = 0.01
epsilon_ifgsm = 0.01
alpha = 0.01
num_iterations = 10

# Convert the input images to floating point tensors
x_train, y_train = train_data["features"], train_data["labels"]
x_train = tf.cast(x_train, tf.float32)

# Define the FGSM function
def fgsm(image, label, epsilon):
    image_tensor = tf.convert_to_tensor(image)
    image_tensor = tf.expand_dims(image_tensor, axis=0)
    image_tensor = tf.cast(image_tensor, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        prediction = model(image_tensor)
        loss = tf.keras.losses.sparse_categorical_crossentropy(label, prediction)
    gradient = tape.gradient(loss, image_tensor)
    signed_grad = tf.sign(gradient)
    perturbation = epsilon * signed_grad
    perturbed_image = image_tensor + perturbation
    perturbed_image = tf.clip_by_value(perturbed_image, 0, 1)
    perturbed_image = tf.squeeze(perturbed_image, axis=0)
    return perturbed_image.numpy()

# Define the IFGSM function
def ifgsm(image, label, epsilon, alpha, num_iterations):
    image_tensor = tf.convert_to_tensor(image)
    image_tensor = tf.expand_dims(image_tensor, axis=0)
    image_tensor = tf.cast(image_tensor, tf.float32)
    perturbed_image = image_tensor
    for i in range(num_iterations):
        with tf.GradientTape() as tape:
            tape.watch(perturbed_image)
            prediction = model(perturbed_image)
            loss = tf.keras.losses.sparse_categorical_crossentropy(label, prediction)
        gradient = tape.gradient(loss, perturbed_image)
        signed_grad = tf.sign(gradient)
        perturbation = alpha * signed_grad
        perturbed_image = perturbed_image + perturbation
        perturbed_image = tf.clip_by_value(perturbed_image, image_tensor - epsilon, image_tensor + epsilon)
        perturbed_image = tf.clip_by_value(perturbed_image, 0, 1)
    perturbed_image = tf.squeeze(perturbed_image, axis=0)
    return perturbed_image.numpy()

# Iterate over the training set and generate adversarial examples
# for image, label in zip(x_train, y_train):
#     # Generate adversarial examples using FGSM
#     perturbed_fgsm = fgsm(image, label, epsilon_fgsm)
    
#     # Generate adversarial examples using IFGSM
#     perturbed_ifgsm = ifgsm(image, label, epsilon_ifgsm, alpha, num_iterations)
    
#    # Save the perturbed images
#     tf.keras.preprocessing.image.save_img(f"perturbed_fgsm_{label}.png", perturbed_fgsm)
#     tf.keras.preprocessing.image.save_img(f"perturbed_ifgsm_{label}.png", perturbed_ifgsm)
    
    
    
import pickle

# Initialize empty lists to store the perturbed images and labels
perturbed_images_fgsm = []
perturbed_images_ifgsm = []
perturbed_labels = []

# Iterate over the training set and generate adversarial examples
for image, label in zip(x_train, y_train):
    # Generate adversarial examples using FGSM
    perturbed_fgsm = fgsm(image, label, epsilon_fgsm)
    
    # Generate adversarial examples using IFGSM
    perturbed_ifgsm = ifgsm(image, label, epsilon_ifgsm, alpha, num_iterations)
    
    # Append the perturbed images and labels to the respective lists
    perturbed_images_fgsm.append(perturbed_fgsm)
    perturbed_images_ifgsm.append(perturbed_ifgsm)
    perturbed_labels.append(label)
   #%% 
# Save the perturbed images and labels to a pickle file
with open("perturbed_images.p", "wb") as f:
    pickle.dump((perturbed_images_fgsm, perturbed_images_ifgsm, perturbed_labels), f)
    


#%%

#%%