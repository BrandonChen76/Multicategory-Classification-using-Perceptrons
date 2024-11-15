import numpy as np
import matplotlib.pyplot as plt

def load_idx1_to_np(path_to_file):

    #open file
    f = open(path_to_file, "rb")

    #Get data based on descriptions 
    magic_number = int.from_bytes(f.read(4), 'big')
    number_of_items = int.from_bytes(f.read(4), 'big')
    
    #read the rest data as unsigned integers
    stuff = np.frombuffer(f.read(), np.uint8).reshape(number_of_items)

    #change the label to the format of this NN's output
    output = np.zeros((number_of_items, 10))
    for i in range(0, number_of_items, 1):
        output[i,stuff[i]] = 1

    output = output.astype(int)

    f.close()
        
    return output

def load_idx3_to_np(path_to_file):

    #open file
    f = open(path_to_file, "rb")

    #Get data based on descriptions 
    magic_number = int.from_bytes(f.read(4), 'big')
    number_of_images = int.from_bytes(f.read(4), 'big')
    number_of_rows = int.from_bytes(f.read(4), 'big')
    number_of_columns = int.from_bytes(f.read(4), 'big')
    
    #read the rest data as unsigned integers
    #normalize input?
    output = np.frombuffer(f.read(), np.uint8).reshape((number_of_images, number_of_rows * number_of_columns)) / 255

    f.close()
        
    return output

#data [1x784]
#weights [784x10]
#prediction [1x10]
def forward(data, weights):

    #[1x10]
    #weight multiplication and summation
    output = np.dot(data, weights)

    #[1x10]
    #fill matrix with zero and change the max of the matrix to be one
    v = np.zeros_like(output)
    v[np.argmax(output)] = 1

    #[1x10]
    #any value greater than or equal to 0 is true and rest are false
    u = (output >= 0).astype(int)

    return output, v, u

#update the weights
def update_weights(weights, learning_rate, expected, u, data):
    return weights + np.outer(data, learning_rate * (expected - u))

#==========================================================================================================================================

tolerance = 0

learning_rate = 1

epoch = 0

epoch_error = []

weights = np.random.rand(784, 10) - 0.5

training_error = 1

n = 50

#training

training_datas = load_idx3_to_np('train-images.idx3-ubyte')
training_classes = load_idx1_to_np('train-labels.idx1-ubyte')

while training_error > tolerance:

    #increment epoch counter
    epoch += 1

    #each epoch
    error = 0
    for i in range(0, n, 1):

        output, prediction_v, prediction_u = forward(training_datas[i], weights)

        #update weights
        weights = update_weights(weights, learning_rate, training_classes[i], prediction_u, training_datas[i])

        #error increase if wrong prediction
        if(not np.array_equal(prediction_v, training_classes[i])):
            error += 1
        
    #error rate
    training_error = error / n
    epoch_error.append(training_error)

plt.plot(epoch_error)
plt.show()

#testing

testing_datas = load_idx3_to_np('t10k-images.idx3-ubyte')
testing_classes = load_idx1_to_np('t10k-labels.idx1-ubyte')

testing_error = 0
error = 0

for i in range(0, testing_datas.shape[0], 1):

    output, prediction_v, prediction_u = forward(training_datas[i], weights)

    #error increase if wrong prediction
    if(not np.array_equal(prediction_v, training_classes[i])):
            error += 1

#error rate
testing_error = error / testing_datas.shape[0]
print("epochs:" , epoch)
print(testing_error)