#%%
from Teacher_model import *
from Student_model import *
from Distillation_model import *
from Data_Preparation import *


# Evaluate the model on the test dataset
student_model.evaluate(x_test, y_test)