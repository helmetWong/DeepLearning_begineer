import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D, MaxPool2D, ZeroPadding2D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model 
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping

################################################################################################
#
# This program continues from the program, "vgg16_intro.py" and "vgg16_SaveModel.py".
# This program demonstrates the architecture of vgg16 and vgg19
# There are two main methods to construct a CNN model. 
#  Method A:  model = Sequential()      
#  Method B:  model = Model(inputs=input, outputs=output) 
#
# I prefer Method A for simple CNN.  I will use Method B for more complicated CNN.
#
# We also apply early stopping in training  
# We can then use "vgg16_LoadModel.py" to load the saved model and predict a single image. 
#
################################################################################################

batch_size = 56    # Using Quadro RTX 5000 with 16G video RAM, "64", "128", "192" and "256" are too large.
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
#    opt = Adam(lr=0.0001)
    opt = SGD(lr=0.00005, momentum=0.9)         
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def VGG16_v2 (input_shape = (224,224,3), classes = num_classes):
    model = Sequential()
    model.add(Input(input_shape, name = "VGG16_input_layer"))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name ="conv-64a") )
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name = "conv-64b"))
    #model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2), name="maxpool-1"))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=128,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name ="conv-128a") )
    model.add(Conv2D(filters=128,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name = "conv-128b"))
    #model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2), name="maxpool-2"))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=256,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name ="conv-256a") )
    model.add(Conv2D(filters=256,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name = "conv-256b"))
    model.add(Conv2D(filters=256,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name = "conv-256c"))
    #model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2), name="maxpool-3"))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=512,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name ="conv-512a") )
    model.add(Conv2D(filters=512, kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name = "conv-512b"))
    model.add(Conv2D(filters=252,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name = "conv-512c"))
    #model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2), name="maxpool-4"))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=512,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name ="conv-512d") )
    model.add(Conv2D(filters=512,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name = "conv-512e"))
    model.add(Conv2D(filters=512,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name = "conv-512f"))
    #model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2), name="maxpool-5"))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu', kernel_initializer='he_uniform', name = 'Dense-4096-1')) 
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', kernel_initializer='he_uniform', name = 'Dense-4096-2'))  
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # compile model
    opt = SGD(lr=0.00005, momentum=0.9)         
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def VGG16_v3 (input_shape = (224,224,3), classes = num_classes):
    input = Input(shape = input_shape, name = "VGG16_input_layer")
    X = Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name ="conv-64a")(input)
    X = Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name = "conv-64b")(X)
    X = MaxPool2D(pool_size=(2,2),strides=(2,2), name="maxpool-1")(X)
    X = Dropout(0.2)(X)

    X = Conv2D(filters=128,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name ="conv-128a")(X)
    X = Conv2D(filters=128,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name = "conv-128b")(X)
    X = MaxPool2D(pool_size=(2,2),strides=(2,2), name="maxpool-2")(X)
    X = Dropout(0.2)(X)

    X = Conv2D(filters=256,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name ="conv-256a")(X)
    X = Conv2D(filters=256,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name = "conv-256b")(X)
    X = Conv2D(filters=256,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name = "conv-256c")(X)
    X = MaxPool2D(pool_size=(2,2),strides=(2,2), name="maxpool-3")(X)
    X = Dropout(0.2)(X)

    X = Conv2D(filters=512,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name ="conv-512a")(X)
    X = Conv2D(filters=512,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name = "conv-512b")(X)
    X = Conv2D(filters=512,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name = "conv-512c")(X)
    X = MaxPool2D(pool_size=(2,2),strides=(2,2), name="maxpool-4")(X)
    X = Dropout(0.2)(X)

    X = Conv2D(filters=512,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name ="conv-512d")(X)
    X = Conv2D(filters=512,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name = "conv-512e")(X)
    X = Conv2D(filters=512,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name = "conv-512f")(X)
    X = MaxPool2D(pool_size=(2,2),strides=(2,2), name="maxpool-5")(X)
    X = Dropout(0.2)(X)

    X = Flatten()(X)
    X = Dense(4096, activation='relu', kernel_initializer='he_uniform', name = 'Dense-4096-1')(X) 
    X = Dropout(0.5)(X)
    X = Dense(4096, activation='relu', kernel_initializer='he_uniform', name = 'Dense-4096-2')(X)  
    X = Dropout(0.5)(X)
    output = Dense(num_classes, activation='softmax')(X)

    model = Model(inputs=input, outputs=output)
    # compile model
    opt = SGD(lr=0.00002, momentum=0.9)         
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def VGG16_v3 (input_shape = (224,224,3), classes = num_classes):
    input = Input(shape = input_shape, name = "VGG16_input_layer")
    X = Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name ="conv-64a")(input)
    X = Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name = "conv-64b")(X)
    X = MaxPool2D(pool_size=(2,2),strides=(2,2), name="maxpool-1")(X)
    X = Dropout(0.2)(X)

    X = Conv2D(filters=128,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name ="conv-128a")(X)
    X = Conv2D(filters=128,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name = "conv-128b")(X)
    X = MaxPool2D(pool_size=(2,2),strides=(2,2), name="maxpool-2")(X)
    X = Dropout(0.2)(X)

    X = Conv2D(filters=256,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name ="conv-256a")(X)
    X = Conv2D(filters=256,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name = "conv-256b")(X)
    X = Conv2D(filters=256,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name = "conv-256c")(X)
    X = Conv2D(filters=256,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name = "conv-256d")(X)
    X = MaxPool2D(pool_size=(2,2),strides=(2,2), name="maxpool-3")(X)
    X = Dropout(0.2)(X)

    X = Conv2D(filters=512,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name ="conv-512a")(X)
    X = Conv2D(filters=512,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name = "conv-512b")(X)
    X = Conv2D(filters=512,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name = "conv-512c")(X)
    X = Conv2D(filters=512,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name ="conv-512d")(X)
    X = MaxPool2D(pool_size=(2,2),strides=(2,2), name="maxpool-4")(X)
    X = Dropout(0.2)(X)

    X = Conv2D(filters=512,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name = "conv-512e")(X)
    X = Conv2D(filters=512,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name = "conv-512f")(X)
    X = Conv2D(filters=512,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name = "conv-512g")(X)
    X = Conv2D(filters=512,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name = "conv-512h")(X)
    X = MaxPool2D(pool_size=(2,2),strides=(2,2), name="maxpool-5")(X)
    X = Dropout(0.2)(X)

    X = Flatten()(X)
    X = Dense(4096, activation='relu', kernel_initializer='he_uniform', name = 'Dense-4096-1')(X) 
    X = Dropout(0.5)(X)
    X = Dense(4096, activation='relu', kernel_initializer='he_uniform', name = 'Dense-4096-2')(X)  
    X = Dropout(0.5)(X)
    output = Dense(num_classes, activation='softmax')(X)

    model = Model(inputs=input, outputs=output)
    # compile model
    opt = SGD(lr=0.00002, momentum=0.9)         
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

#model = VGG16_v1(input_shape = [img_width,img_height,3], classes = num_classes)
#model = VGG16_v2(input_shape = [img_width,img_height,3], classes = num_classes)
model = VGG16_v3(input_shape = [img_width,img_height,3], classes = num_classes)
#model = VGG19_v2(input_shape = [img_width,img_height,3], classes = num_classes)


model.summary()

############################################################################ 
# Apply early stopping commands

monitor = EarlyStopping(monitor='val_loss', min_delta =0.001, patience = 5, verbose = 0,
                       mode = 'auto', restore_best_weights = True )
epochs = 100 
history = model.fit(train_ds, 
                    steps_per_epoch = len(train_ds),  #batch_size
                    epochs = epochs,
                    validation_data = val_ds,         #batch_size
                    validation_steps = len(val_ds),
                    callbacks=[monitor]
                    )

############################################################################
# Save a model

model.save("vgg16_v3_t1.h5")

############################################################################
# Save history of training

df = pd.DataFrame(history.history)
filename = 'history_vgg16_v3_t1.csv'
with open (filename, mode ='w') as f:
    df.to_csv(f)


