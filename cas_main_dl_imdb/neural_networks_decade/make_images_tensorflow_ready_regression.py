# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 10:15:54 2020

Get proper image data to feed in a tensorflow neural network (train, validation and test)

@author: simon
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from PIL import Image
import os
import pathlib
import tensorflow as tf

#Set directory where the images are stored
data_dir_overall = "C:/Users/simon/Documents/ZHAW Data Science/CAS_Machine_Intelligence/Deep_Learning/Project/categorised_pics_decade_regression"

###Read in train data
data_dir = pathlib.Path(data_dir_overall + "/train")

#Get decades (classes)
class_names = np.array([item.name for item in data_dir.glob('*')])
class_names.astype(int)

#Image count
image_count = len(list(data_dir.glob('*/*.jpg')))
print("There are",image_count,"movie posters in the training set.")

#Define way to load images - already normalize here -> Batch size can be adapted
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

BATCH_SIZE = 100
IMG_HEIGHT = 150
IMG_WIDTH = 101
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)

#Load images 
train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode = "sparse",
                                                     classes = list(class_names))

print("finished preparing data of training set, stored as train_data_gen")


###Read in validation data
data_dir = pathlib.Path(data_dir_overall + "/validate")

#Image count
image_count = len(list(data_dir.glob('*/*.jpg')))
print("There are",image_count,"movie posters in the validation set.")

STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)

#Load images 
val_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode = "sparse",
                                                     classes = list(class_names))

print("finished preparing data of validation set, stored as val_data_gen")

###Read in test data
data_dir = pathlib.Path(data_dir_overall + "/test")

#Image count
image_count = len(list(data_dir.glob('*/*.jpg')))
print("There are",image_count,"movie posters in the test set.")

STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)

#Load images 
test_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode = "sparse",
                                                     classes = list(class_names))

print("finished preparing data of test set, stored as test_data_gen")

#Show 12 random movie posters of every set

def show_batch(image_batch, label_batch):
  plt.figure(figsize=(15,8))
  for n in range(12):
      ax = plt.subplot(2,6,n+1)
      plt.imshow(image_batch[n])
      plt.title(int(label_batch[n]+1970))
      plt.axis('off')

print("Show 12 random posters from training, validation and test set")
image_batch, label_batch = next(train_data_gen)
show_batch(image_batch, label_batch)

image_batch, label_batch = next(val_data_gen)
show_batch(image_batch, label_batch)

image_batch, label_batch = next(test_data_gen)
show_batch(image_batch, label_batch)

