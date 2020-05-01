# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 20:26:12 2020

First CNN neural Network with dropout to predict the decade of a movie poster

@author: simon
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten , Activation, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras import optimizers
import tensorflow as tf
import os
import matplotlib.pyplot as plt
#Set working directory
os.chdir("C:/Users/simon/Documents/ZHAW Data Science/CAS_Machine_Intelligence/Deep_Learning/Project/")

#Get train, validation and test data
exec(open("Scripts/make_images_tensorflow_ready.py").read())

#Size of movie poster
IMG_HEIGHT = 150
IMG_WIDTH = 101

#Model from cifar example with dropout
model  =  Sequential()

model.add(Convolution2D(16,(3,3),activation="relu",padding="same",input_shape=(150,101,3)))
model.add(Convolution2D(16,(3,3),activation="relu",padding="same"))
model.add(MaxPooling2D((2,2)))

model.add(Convolution2D(32,(3,3),activation="relu",padding="same"))
model.add(Convolution2D(32,(3,3),activation="relu",padding="same"))
model.add(MaxPooling2D((2,2)))

model.add(Convolution2D(16,(3,3),activation="relu",padding="same"))
model.add(Convolution2D(16,(3,3),activation="relu",padding="same"))
model.add(MaxPooling2D((2,2)))


model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(300))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(100))
model.add(Activation('relu'))

model.add(Dense(5))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

#Train the Model with 10 epochs
epochs=10

history=model.fit(train_data_gen, 
                  epochs=epochs,
                  verbose=1, 
                  validation_data=(val_data_gen)
                 )

#Show the results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

#Check with test dataset
results = model.evaluate(test_data_gen)
print('test loss, test acc:', results)
print(results)

#Plot Predictions
pred = model.predict(test_data_gen)

classes = []

for i in range(0,10000):
    classes.append(np.argmax(pred[i]))

plt.figure(figsize=(14,6))
classes_names = ("1970-1979","1980-1989","1990-1999","2000-2009","2010-2019")
y_pos = np.arange(len(classes_names))
classes_count = [classes.count(2),classes.count(3),classes.count(4),classes.count(0),classes.count(1)]
plt.bar(y_pos, classes_count, align='center', alpha=0.5)
plt.xticks(y_pos, classes_names)
plt.ylabel('Amount')
plt.title('Predictons of test set')
plt.show()

plt.figure(figsize=(14,6))
classes_names = ("1970-1979","1980-1989","1990-1999","2000-2009","2010-2019")
y_pos = np.arange(len(classes_names))
classes_count = [2897,3253,3453,6623,13625]
plt.bar(y_pos, classes_count, align='center', alpha=0.5)
plt.xticks(y_pos, classes_names)
plt.ylabel('Amount')
plt.title('Distribution of decades in training set')
plt.show()

#model.save_weights('./weights_regression_model/weights_cnn_model')
#model.load_weights('./weights_regression_model/weights_cnn_model')