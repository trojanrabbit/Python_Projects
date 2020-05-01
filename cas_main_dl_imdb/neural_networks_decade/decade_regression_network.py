# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 20:26:12 2020

Decade regression network to predict the decade of a movie poster

@author: simon
"""


import numpy as np
import urllib
import os
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten , Activation, Dropout, Input, Concatenate
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras import optimizers
tfd = tfp.distributions

#Set working directory
os.chdir("C:/Users/simon/Documents/ZHAW Data Science/CAS_Machine_Intelligence/Deep_Learning/Project/")

#Get train, validation and test data
exec(open("Scripts/make_images_tensorflow_ready_regression.py").read())

#Size of movie poster
IMG_HEIGHT = 150
IMG_WIDTH = 101

#Distribution of target variable (first batch)
plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
plt.hist(train_data_gen[0][1]+1970,bins=20)
plt.title("Age dist train")
plt.subplot(1,2,2)
plt.hist(val_data_gen[0][1]+1970,bins=20)
plt.title("Age dist val")
plt.show()

#Create Model
kernel_size = (3, 3)
pool_size = (2, 2)

def NLL(y, distr):
  return -distr.log_prob(y) 

def my_dist(params): 
  return tfd.Normal(loc=params[:,0:1], scale=1e-3 + tf.math.softplus(0.05 * params[:,1:2]))# both parameters are learnable

inputs = Input(shape=(150,101,3))
x = Convolution2D(16,kernel_size,padding='same',activation="relu")(inputs)
x = Convolution2D(16,kernel_size,padding='same',activation="relu")(x)
x = MaxPooling2D(pool_size=pool_size)(x)

x = Convolution2D(32,kernel_size,padding='same',activation="relu")(x)
x = Convolution2D(32,kernel_size,padding='same',activation="relu")(x)
x = MaxPooling2D(pool_size=pool_size)(x)

x = Convolution2D(32,kernel_size,padding='same',activation="relu")(x)
x = Convolution2D(32,kernel_size,padding='same',activation="relu")(x)
x = MaxPooling2D(pool_size=pool_size)(x)

x = Flatten()(x)
x = Dense(500,activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(50,activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(2)(x)
dist = tfp.layers.DistributionLambda(my_dist)(x) 

model_flex = Model(inputs=inputs, outputs=dist)
model_flex.compile(tf.keras.optimizers.Adam(), loss=NLL)

#Train the Model with 10 epochs
epochs=10

# train the model
history=model_flex.fit(train_data_gen,
                    epochs=epochs,
                    verbose=1, 
                    validation_data=val_data_gen)
                  
#Show the results
model_mean = Model(inputs=inputs, outputs=dist.mean())
model_sd = Model(inputs=inputs, outputs=dist.stddev())

loss=history.history['loss']
val_loss=history.history['val_loss']

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.show()

#Check with test dataset
results = model_flex.evaluate(test_data_gen,verbose=1)
print('nll:', results)

#Create predictions
model_mean = Model(inputs=inputs, outputs=dist.mean())
pred = model_mean.predict(test_data_gen, verbose=1)

#Show predictions of 15 Poster

plt.figure(figsize=(20,20))
for i in range(0,15):
    plt.subplot(3,5,i+1)
    plt.imshow(test_data_gen[0][0][i])
    plt.title("pred:"+ str(int(pred[i][0]+1970)) + " true:"+ str(int(test_data_gen[0][1][i]+1970)), fontsize=18, fontweight="bold")

    
pred_new = pred + 1970  
plt.hist(pred_new, bins=20, range=(1970,2020))

#Save weights
#model_flex.save_weights('./weights_regression_model/weights_regression_model')

#Load weights
#model_flex.load_weights('./weights_regression_model/weights_regression_model')