import numpy as np

training_set = np.array([[0,0],
                         [0,1],
                         [1,0],
                         [1,1]])

labels = np.array([[0,
                    1,
                    1,
                    1]])

labels = labels.reshape(4,1)

    
#Hyperparameters
np.random.seed(42)
weights = np.random.rand(2,1)
bias = np.random.rand(1)
LR = 0.05
EPOCHS = 1000

#Activation functions
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))
  
  

