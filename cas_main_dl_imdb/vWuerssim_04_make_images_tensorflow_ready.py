# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 10:15:54 2020

Get proper image data to feed in a tensorflow neural network (train, validation and test)

@author: simon
"""

##############---- import packages ----##############
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import tensorflow as tf


##############---- Functions ----##############
#---- load image data
def loadImages(data_sets, b_size, px_h, px_w): 
    img_data = {}
    for key, path in data_sets.items():
        #---- image count
        image_count = len(list(path.glob('*/*.jpg')))
        print(f"There are {image_count} movie posters in the {key} set.")
        print(f"generating dateset from images for {key} set...")
        #---- get classes
        class_names = np.array([item.name for item in path.glob('*')])
        #---- normalize
        image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        #---- load images
        img_data[key] = image_generator.flow_from_directory(directory=str(path),
                                                     batch_size= b_size,
                                                     shuffle=True,
                                                     target_size=(px_h, px_w),
                                                     classes = list(class_names))        
    return img_data

#---- plot random images
def showBatch(image_batch, label_batch, data_set):
    class_names = np.array([item.name for item in data_set.glob('*')])
    plt.figure(figsize=(20,12))
    for n in range(18):
        ax = plt.subplot(3,6,n+1)
        plt.imshow(image_batch[n])
        plt.title(class_names[label_batch[n]==1][0].title())
        plt.axis('off')


##############---- Main ----##############
#---- define inputs
img_height = 150
img_width = 101
batch_size = 128            
# directory where images are stored
data_dir = "data/dl_imdb_pics/pics_genre"

#---- dictionary with paths to data sets
data_dir_sets = {"train": pathlib.Path(data_dir + "/train"), 
            "validate": pathlib.Path(data_dir + "/validate"), 
            "test": pathlib.Path(data_dir + "/test")}

#---- load images
img_data = loadImages(data_dir_sets, batch_size, img_height, img_width)

#---- get datasets
train_data_gen = img_data['train']
val_data_gen = img_data['validate']
test_data_gen = img_data['test']

#---- Show 18 random movie posters of every set
# train
image_batch, label_batch = next(img_data["train"])
showBatch(image_batch, label_batch, data_dir_sets["train"])

# test
image_batch, label_batch = next(img_data["test"])
showBatch(image_batch, label_batch, data_dir_sets["test"])

# validation
image_batch, label_batch = next(img_data["validate"])
showBatch(image_batch, label_batch, data_dir_sets["validate"])