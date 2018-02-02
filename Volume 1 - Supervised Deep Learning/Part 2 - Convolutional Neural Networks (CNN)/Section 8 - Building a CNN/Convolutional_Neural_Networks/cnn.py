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

#Add another convulutional layer to further increase accuracy
classifier.add(Convolution2D(32,(3,3), activation='relu'))
#Max Pooling Step,
classifier.add(MaxPooling2D(pool_size = (2,2)))
#Flattening step
classifier.add(Flatten())
#Step 4 Fully Connect
classifier.add(Dense(activation='relu', units = 128))
#One node expects dog or cat on this layer 
classifier.add(Dense(activation='sigmoid', units = 1 ))

#Compile the cnn
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Image preprocessing
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=64,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=64,
        class_mode='binary')

import tensorflow as tf

with tf.device('/gpu:1'):
    classifier.fit_generator(training_set,
                         steps_per_epoch = (8000/32),
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = (2000/32),
                          use_multiprocessing=True, workers=8)



