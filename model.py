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
  
  
#Training the model
for epoch in range(EPOCHS):
    inputs = training_set
    z = np.dot(inputs, weights) + bias
    prediction = sigmoid(z)
    error = prediction - labels
    
    print(f"Epoch {epoch+1}/{EPOCHS} \n  Error: \n{error}\n")
    change = error * sigmoid_derivative(prediction)
    inputs = training_set.T
    weights = weights - LR*np.dot(inputs, change)
    for num in change:
        bias = bias - LR*num
        
        
#Feed forward
inputs = training_set
XW = np.dot(inputs, weights) + bias
prediction = sigmoid(XW)

#Error
error = prediction - labels


change = error * sigmoid_derivative(prediction)
inputs = training_set.T
weights = weights - LR*np.dot(inputs, change)

for num in change:
    bias = bias - LR*num
    
    

