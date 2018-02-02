# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()

#Adding convulution
classifier.add(Convolution2D(32,(3,3),input_shape=(64,64,3), activation='relu'))
#Max Pooling Step,
classifier.add(MaxPooling2D(pool_size = (2,2)))
#Flattening step
classifier.add(Flatten())
#Step 4 Fully Connect
classifier.add(Dense(activation='relu', units = 128))
#One node expects dog or cat on this layer 
classifier.add(Dense(activation='sigmoid', units = 1 ))





