# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 09:32:02 2020

@author: Trojan Rabbit
"""

##############---- import packages ----##############
from sklearn.metrics import confusion_matrix
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten , Activation, Dropout
#from tensorflow.keras.utils import to_categorical 
from tensorflow.keras import optimizers

##############---- Main ----##############
#---- general input
img_height = 150
img_width = 101
color_channels = 3
nr_of_classes = 21
batch_size = 64
epochs = 10


#------------------------- fcNN -------------------------#
########## 1
#---- input
flat_size = img_height * img_width * color_channels

#---- create model
model = Sequential()

model.add(Dense(150, batch_input_shape = (None, flat_size)))
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(nr_of_classes))
model.add(Activation('softmax'))

#---- compile model and intitialize weights
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'sgd',
              metrics = ['accuracy'])

model.summary()

#---- train the model
history=model.fit(train_data_flat, train_label_enc, 
                  batch_size = batch_size, 
                  epochs = epochs,
                  verbose = 2, 
                  validation_data = (val_data_flat, val_label_enc))


########## 2
#---- input
flat_size = img_height * img_width * color_channels

#---- create model
model = Sequential()

model.add(Dense(200, batch_input_shape = (None, flat_size)))
model.add(Activation('relu'))
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(nr_of_classes))
model.add(Activation('softmax'))

#---- compile model and intitialize weights
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'sgd',
              metrics = ['accuracy'])

model.summary()

#---- train the model
history=model.fit(train_data_flat, train_label_enc, 
                  batch_size = batch_size, 
                  epochs = epochs,
                  verbose = 2, 
                  validation_data = (val_data_flat, val_label_enc))

#------------------------- CNN -------------------------#
#---- input
kernel_size = (3, 3)
input_shape =  (img_height, img_width, color_channels)
pool_size = (2, 2)

#---- create model
model = Sequential()

model.add(Convolution2D(filters = 16, kernel_size = kernel_size, padding = 'same', input_shape = input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(filters = 16, kernel_size = kernel_size, padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = pool_size))

model.add(Convolution2D(filters = 32, kernel_size = kernel_size, padding = 'same'))
model.add(Activation('relu'))
model.add(Convolution2D(filters = 32, kernel_size = kernel_size, padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = pool_size))

model.add(Convolution2D(filters = 64, kernel_size = kernel_size, padding = 'same'))
model.add(Activation('relu'))
model.add(Convolution2D(filters = 64, kernel_size = kernel_size, padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = pool_size))

model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(50))
model.add(Activation('relu'))

model.add(Dense(nr_of_classes))
model.add(Activation('softmax'))

model.summary()

#---- compile model and intitialize weights
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

#---- train the model
history = model.fit(train_data_n, train_label_enc,                
                    batch_size = batch_size, 
                    epochs = epochs,
                    verbose = 2,
                    validation_data = (val_data_n, val_label_enc)) 


##########################################

# plot the development of the accuracy and loss during training
plt.figure(figsize=(12,4))
plt.subplot(1,2,(1))
plt.plot(history.history['accuracy'],linestyle='-.')
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='lower right')
plt.subplot(1,2,(2))
plt.plot(history.history['loss'],linestyle='-.')
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper right')
plt.show()

# Prediction on the test set
pred=model.predict(test_data_n)
print(confusion_matrix(np.argmax(test_label_enc,axis=1),np.argmax(pred,axis=1)))
acc_fc = np.sum(np.argmax(test_label_enc,axis=1)==np.argmax(pred,axis=1))/len(pred)
print("Acc = " , acc_fc)
