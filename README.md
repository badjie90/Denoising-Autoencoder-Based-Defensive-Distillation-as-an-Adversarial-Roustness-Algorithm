# Defensive-Distillation-as-an-Adversarial-Roustness-Algorithm

Contained within this repository is a defensive distillation project written in the Python programming language. The architecture of this project is such that it is adaptable and can be leverage to robustify any deep neural network, with any image dataset.

To make use of this project, a sequential execution of the following files is necessary: Data_Preparation.py, Teacher_model.py, Distillation_model.py, Student_model.py, Evaluate_distilled_model_1.py, Ensemble_adversarial_attacks.py, and Evaluate_distilled_model_2.py.

The original concept of defensive distillation as an adversarial robustness algorithm was first introduced by Nicolas Papernot of The Pennsylvania State University, in a paper accessible via this [link](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Distillation+as+a+Defense+to+Adversarial+Perturbations+against+Deep+Neural+Networks&btnG=). However, we have made a notable advancement to this method by addressing the issue in which attackers could reverse-engineer both models to discover fundamental exploits. Moreover, we have implemented training data filtering techniques to mitigate poisoning attacks, in which the initial training database is corrupted by a malicious actor.

To expound further, our approach consists of generating a teacher model that is trained on the initial training dataset. The teacher model's predictions are then used to create a new dataset, which is used to train the student model. This process of distillation provides a more robust and secure model that is less susceptible to adversarial attacks. Additionally, we introduce a novel technique that ensembles multiple adversarial attacks to assess the model's resilience, providing a more comprehensive evaluation.

Overall, this repository offers a cutting-edge approach to improving the security of deep neural networks, addressing issues that have long plagued the field. With this project, one can easily integrate our defensive distillation methodology into their existing models, enhancing their robustness and mitigating the risks of adversarial attacks.




## To download and convert the dataset into pickle file:

### run the following command in the project directory

$ ./downloadDataset.sh This will download the zip file and unzip in the current directory as GTSRB. The data formatting of images is given in Readme-Images.txt

To generate pickle files using GTSRB dataset change the parameters according to the requirement

dataGen = datasetGenerator(nClasses=5, nTrainSamples=800, nTestSamples=100, nValidateSamples=100,imageSize=[28, 28])

nClasses - no of classes nTrainSamples - no of Training samples nTestSamples - no of test samples nValidateSamples - no of validation samples imageSize - size of 2D image matrix to be resized into. $ python datasetGenerator.py

This will generate the following files

├── info.txt

├── test.p

├── train.p

└── validate.p
