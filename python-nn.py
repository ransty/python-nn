import numpy as np

# The program creates an neural network that simulates the exclusive OR function with two inputs and one output

# Sigmoid function
def nonlin(x, deriv=False):
    if(deriv==True):
        return (x*(1-x))
    return 1/(1+np.exp(-x))

# Input Matrix (each row is a different training example, each column is a different training neuron)
X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])

# Output data (4 examples, one output neuron each)
y = np.array([[1],
            [0],
            [1],
            [0]])

# Seed for random generation
np.random.seed(1)

# Synapses (connections between each neuron in one layer to the next layer)
syn0 = 2*np.random.random((3, 4)) - 1 # 3x4 matrix of weights (random weights)
syn1 = 2*np.random.random((4, 1)) - 1 # 4x1 matrix of weights (random weights)

# training time
for j in range(100000):
    aa = X
    ff = nonlin(np.dot(aa, syn0))
    ee = nonlin(np.dot(ff, syn1)) # predictions (sigmoid)

    # backpropagation (chain rule)
    ee_error = y - ee
    if(j % 10000) == 0:
        print("Error: " + str(np.mean(np.abs(ee_error))))

    ee_delta = ee_error*nonlin(ee, deriv=True) # used to reduce error rate

    ff_error = ee_delta.dot(syn1.T) # how much layer one contributed to the error

    ff_delta = ff_error * nonlin(ff, deriv=True)

    # update weights (gradient descent)
    syn0 += aa.T.dot(ff_delta)
    syn1 += ff.T.dot(ee_delta)

print("Output after training")
print(ee)
