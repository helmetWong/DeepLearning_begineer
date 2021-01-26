import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D, MaxPool2D, ZeroPadding2D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model 
from tensorflow.keras.optimizers import Adam, SGD


################################################################################################
#
# This program continues from the program, "vgg16_intro.py" which it does not contain save function.
# This program can save the model of VGG16 and weights into a h5 format, 
# and the model in the program, "vgg16_LoadModel.py" and predict a single image. 
#
################################################################################################


batch_size = 128  # Using Quadro RTX 5000 with 16G video RAM, "192" and "256" are too large.
                    # Reduce the batch_size if the vidoe RAM is not enough
img_width = 224     # (224, 224, 3) standard input size for VGG model
img_height = 224 

# Download the "flower_photos.tgz" from below website, and put into "date_dir"
# /datasets/flowers_photos/daisy/
# /datasets/flowers_photos/dandelion/
# /datasets/flowers_photos/roses/
# /datasets/flowers_photos/sunflowers/
# /datasets/flowers_photos/tulips/
# https://www.tensorflow.org/tutorials/images/classification
# https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz

data_dir = "E:/datasets/flower_photos/"


############################################################################
# load image from data_dir in batches

datagen = ImageDataGenerator(rescale = 1./255,
                             validation_split = 0.2)  #set validation split
classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
num_classes = len(classes)

train_ds = datagen.flow_from_directory(directory=data_dir, 
                                       target_size= (img_width, img_height),
                                       classes = classes,
                                       class_mode = 'categorical',
                                       shuffle = True,
                                       batch_size = batch_size, 
                                       subset='training')

val_ds = datagen.flow_from_directory(directory = data_dir,
                                     target_size= (img_width, img_height),
                                     classes = classes,
                                     class_mode = 'categorical',
                                     shuffle = False,
                                     batch_size = batch_size, 
                                     subset='validation')


############################################################################
# Use VGG16 with pre-trained weights

def VGG16_v1(input_shape = (224,224,3), classes = num_classes):

    model = VGG16(include_top=False, input_shape= input_shape, weights='imagenet')
    X = model.output
    X = Flatten()(X)
    X = Dense(4096, activation='relu', kernel_initializer='he_uniform', name = 'Dense-4096-1')(X) 
    X = Dense(4096, activation='relu', kernel_initializer='he_uniform', name = 'Dense-4096-2')(X)     
    output = Dense(num_classes, activation='softmax', name = 'output_layer')(X)

    model = Model(inputs=model.inputs, outputs=output)

    opt = SGD(lr=0.00005, momentum=0.9)         
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = VGG16_v1(input_shape = [img_width,img_height,3], classes = num_classes)

model.summary()

epochs = 15     
history = model.fit(train_ds, 
                    steps_per_epoch = len(train_ds),  #batch_size
                    epochs = epochs,
                    validation_data = val_ds,   #batch_size
                    validation_steps = len(val_ds)
                    )

############################################################################
# Save a model
model.save("vgg16_t2.h5")


############################################################################
# Save history of training

df = pd.DataFrame(history.history)
filename = 'history_vgg16_t2.csv'
with open (filename, mode ='w') as f:
    df.to_csv(f)


