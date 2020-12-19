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

np.random.seed(42)
weights = np.random.rand(3,1)
bias = np.random.rand(1)
lr = 0.05



