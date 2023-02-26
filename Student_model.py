#%%

from Data_Preparation import *
from Teacher_model import *
from Distillation_model import *



# Generate a distilled dataset using defensive distillation
X_train_distilled, y_train_distilled = defensive_distillation(x_train, y_train)

#%%

# Train a Student model on the distilled dataset

#Student_model = "model_distilled"

student_model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(43)
])

student_model.compile(optimizer='adam',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])

student_model.fit(X_train_distilled, y_train_distilled, epochs=10)

# Save the model.
student_model.save("/home/bakary/Desktop/My_GitHub/Defensive_distillation/Defensive-Distillation-as-an-Adversarial-Roustness-Algorithm/model.h5")