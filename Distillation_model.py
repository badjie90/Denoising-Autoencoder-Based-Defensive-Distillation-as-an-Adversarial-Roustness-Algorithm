#%%
#import Teacher_model
import tensorflow as tf
from Teacher_model import teacher_model
#from Student_model import *
from Data_Preparation import *
from Denoising_autoencoder import x_train_clean

# Define the defensive distillation function
def defensive_distillation(X, y, T=5, alpha=0.1):
    """
    Generate a distilled dataset using defensive distillation
    """
    logits = teacher_model(X) / T
    labels = tf.math.softmax(logits)
    labels_distilled = tf.math.pow(labels, 1 / alpha)
    labels_distilled = labels_distilled / tf.math.reduce_sum(labels_distilled, axis=1, keepdims=True)
    return X, labels_distilled



# #Train the teacher model on the original dataset
# teacher_model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

# teacher_model.fit(x_train, y_train, epochs=10)

# Use the clean training data to train the teacher model using defensive distillation

X, y = defensive_distillation(x_train_clean, y_train)


teacher_model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

teacher_model.fit(X, y, epochs=20)



  

   