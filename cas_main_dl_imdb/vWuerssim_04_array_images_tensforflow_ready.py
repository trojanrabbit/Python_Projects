# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 10:02:49 2020

@author: Trojan Rabbit
"""

##############---- import packages ----##############
from PIL import Image
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


##############---- Functions ----##############
def getImages(img_dir):
    img_dataset = {} # dict fuer image datasets test, train, validate
    img_classset = {} # dict fuer image classes
    data_sets = ['train', 'validate', 'test']
    count = 0
    for data_set in data_sets:
        print(f"get images for {data_set}...")
        print('---------------------------------')
        img_data = []
        img_class = []
        for path in glob.glob(img_dir + data_set + '/*/*.jpg', recursive = True): 
            img = Image.open(path)                        
            try:
                np.reshape(img, (1, 150, 101, 3)) # wenn Fehler, dann nicht RGB -> Bild verwerfen -> except-Block
                img_arr = np.array(img)
                img_data.append(img_arr)
                img_class.append(os.path.basename(os.path.dirname(path)))
            except:                 
                #img = Image.new("RGB", img.size)                
                #img_arr = np.array(img)
                #img_data.append(img_arr)
                #img_class.append(os.path.basename(os.path.dirname(path)))
                count += 1
                print(f'following image is not RGB and therefore not included: {path}')
        img_data_arr = np.array(img_data)
        img_dataset[data_set] = img_data_arr # img dataset zu dict hinzufuegen
        img_class_arr = np.array(img_class)
        img_classset[data_set] = img_class_arr # img classes zu dict hinzufuegen
    print(f'total images not included: {count}')
    print("image load abgeschlossen")
    return img_dataset['train'], img_classset['train'], img_dataset['validate'], img_classset['validate'], img_dataset['test'], img_classset['test']

def oneHotEncoding(data):
    # https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(data)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse = False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded

def plotImages(img_data, label_data, row_filter):
    if row_filter != None:
        ind = np.where(label_data == row_filter)
        label_data = label_data[ind]
        img_data = img_data[ind]    
    ind = np.random.choice(img_data.shape[0], size = 18, replace = False)
    img_data_s = img_data[ind]
    label_data_s = label_data[ind]
    plt.figure(figsize=(20,12))
    for n in range(len(ind)):
        ax = plt.subplot(3,6,n+1)
        plt.imshow(img_data_s[n])
        plt.title(label_data_s[n].title())
        plt.axis('off')


##############---- Main ----##############
#---- define inputs
img_height = 150
img_width = 101
color_channels = 3
# directory where images are stored
data_dir = "data/dl_imdb_pics/pics_genres_strat1k/" # mit / am Ende
      
#---- load image data
train_data, train_label, val_data, val_label, test_data, test_label = getImages(data_dir)
del data_dir

#---- normalise image data
train_data_n = train_data / 255
del train_data
val_data_n = val_data / 255
del val_data
test_data_n = test_data / 255
del test_data

#---- one hot encoding of labels
train_label_enc = oneHotEncoding(train_label)
val_label_enc = oneHotEncoding(val_label)
test_label_enc = oneHotEncoding(test_label)

#---- we need vectors for fcNN  (only for fcNN)
flat_size = img_height * img_width * color_channels
train_data_flat = train_data_n.reshape([train_data_n.shape[0], flat_size])
val_data_flat = val_data_n.reshape([val_data_n.shape[0], flat_size])
test_data_flat = test_data_n.reshape([test_data_n.shape[0], flat_size])

#---- plot 18 random images
# choose from a class if desired
np.unique(train_label)
row_filter = '80' # class (decade, genre) or None
plotImages(train_data_n, train_label, row_filter)
plotImages(val_data_n, val_label, row_filter)
plotImages(test_data_n, test_label, row_filter)

del train_label, val_label, test_label
