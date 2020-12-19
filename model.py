import numpy as np

training_set = np.array([[0,1,0],
                         [0,0,1],
                         [1,0,0],
                         [1,1,0],
                         [1,1,1],
                         [0,1,1],
                         [0,1,0]])

labels = np.array([[1,
                    0,
                    0,
                    1,
                    1,
                    0,
                    1]])


labels = labels.reshape(7,1)



#Hyperparameters
np.random.seed(42)
weights = np.random.rand(3,1)
bias = np.random.rand(1)
lr = 0.05




#Activation functions
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))
 

#Training the model
for epoch in range(30000):
    inputs = training_set
    XW = np.dot(inputs, weights) + bias
    z = sigmoid(XW)
    error = z - labels
    print(error.sum())
    dcost = error
    dpred = sigmoid_derivative(z)
    z_del = dcost * dpred
    inputs = training_set.T
    weights = weights - lr*np.dot(inputs, z_del)
    for num in z_del:
        bias = bias - lr*num

inputs = training_set




#Feed forward
XW = np.dot(inputs, weights) + bias
z= sigmoid(XW)

error = z - labels

slope = inputs * dcost * dpred

dcost = error
dpred = sigmoid_derivative(z)
z_del = dcost * dpred
inputs = training_set.T
weights = weights - lr*np.dot(inputs, z_del)

for num in z_del:
    bias = bias - lr*num
    
    
for i in training_set:
    single_pt = np.array(i)
    result = sigmoid(np.dot(single_pt, weights) + bias)
    print(f'{result}\nRounded result: {np.around(result)}\n')
  
    
    
    

