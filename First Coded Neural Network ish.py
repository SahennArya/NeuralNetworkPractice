import numpy as np

#sigmoid function - normalises to range (0,1)
#also returns derivative of the function when deriv is true
#the derivative of sigmoid is used to make greater adjustments to weights that have a greater error and vice versa
def sigmoid (x,deriv=False):
        if deriv == True:
                return x * (1 - x)
        return 1 / (1 + np.exp(-x))

#training inputs
training_set_inputs = np.array([[0, 0, 1, 1],
                                [1, 1, 1, 0],
                                [1, 0, 1, 1],
                                [0, 1, 1, 1]])
#what we want the network to return
training_set_outputs = np.array([[0, 1, 1, 0]]).T

#keeps the random numbers set each time
np.random.seed(1)

#synapse between neuron
syn0 = 2*np.random.random((4,1))-1

#training the model 10,000 times. Less makes it untrained, more makes it overfitted to this dataset - wont be good on new test data
for i in range (10000):
    #input layer
    l0 = training_set_inputs
    #output layer
    l1 = sigmoid(np.dot(l0,syn0))

    #Calculating the error (difference betweeen output and desired output)
    l1_error = training_set_outputs - l1
    #Multiply the error by the input and again by the gradient of the Sigmoid curve.
    # This means less confident weights are adjusted more.
    # This means inputs, which are zero, do not cause changes to the weights.
    l1_delta = l1_error * sigmoid(l1,True)

    weightAdjustment = np.dot(l0.T,l1_delta)
    #calculating the adjustments required to the weights after each iteration and changing the synapse accordingly
    syn0 += weightAdjustment

print(l1)