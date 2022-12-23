import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten

#Check tf version
print("Version: ", tf.__version__)

#Get MNIST dataset.
#Image classification of digits.

#train and test sets
(x_train,y_train),(x_test,y_test) = mnist.load_data()

#Shape of each training example is a tensor
#of (1,28,28)
print("Shape of one sinlge x:",x_train[0:1,:].shape)
print("x_train shape: ", x_train.shape)


#shape of each output y (1,). One single number 
print("Shape of one single y:",y_train[0:1].shape)
print("First 10 digits:",y_train[0:10])
print("y_train shape: ", y_train.shape)

#Show examples
#Images 
example = 35
imgplot = plt.imshow(x_train[example,:])
plt.show()

#Normalize data between 0 and 1
x_train = x_train/255.0

#Create Neural Network using keras :D (very easy)
model = tf.keras.models.Sequential(
    [Flatten(),#Unroll the images from (1,28,28) shape to (1,784) shape each one.
    Dense(16,activation = 'relu'), #16 neurons with relu activation function.
    Dense(10,activation='softmax')] # 10 classes (0,1,...9), we need 10 output neurons. 
)

#Compile model. Add loss function, optimization algo and evaluation metric(s).
model.compile(
    optimizer='adam', #more advanced version of gradient descent.
    loss='sparse_categorical_crossentropy', #for classification
    metrics = ['accuracy'])

input_shape = (None,28,28) #the shape of the input data (batch_size,data shape)
model.build(input_shape)
print("Model summary: ")
model.summary()

#Change to activate/deactivate training
TRAIN = False
if TRAIN:
    #Train the model.
    print("Training ...")
    model.fit(x_train,y_train,epochs = 4, batch_size = 32)

#Normalize data between 0 and 1
x_test = x_test/255.0

print("Evaluating on test data...")
#Eval the model on test set.
model.evaluate(x_test,y_test,batch_size = 32)


#Prediction
#Add batch dimention to example (1,28,28)
pred = model(x_train[example][None,:,:])
#Can make batch predictions. Several predictions at one 
#100 predictions :D
#pred = model(x_train[10:110])
print("Neural net output:", pred)
print("Digit predicted: ",int(tf.math.argmax(pred,axis = 1)))

