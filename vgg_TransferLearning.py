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
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model 
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping

################################################################################################
#
# This program continues from the program, "vgg16_intro.py" and "vgg16_SaveModel.py".
# This program demonstrates data augmentations, transfer learning and exponentialDecay learning rate.  
# We use both vgg16 and vgg19 
#
# We also apply early stopping in training  
# We can then use "vgg16_LoadModel.py" to load the saved model and predict a single image. 
#
################################################################################################

batch_size = 256  # Using Quadro RTX 5000 with 16G video RAM.
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
# Data augmentation

datagen = ImageDataGenerator(rescale = 1./255,
                             rotation_range=40,
                             width_shift_range = 0.2,
                             height_shift_range = 0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             vertical_flip=True,
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
# Use VGG16 with pre-trained weights, and apply transfer learning 

def VGG16_v1(input_shape = (224,224,3), classes = num_classes):

    model = VGG16(include_top=False, input_shape= input_shape, weights='imagenet')
    for layer in model.layers:
        layer.trainable = False  #15 = True
    
    X = model.layers[-5].output
    X = Conv2D(filters=512,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name = "conv-512a")(X)
    X = Conv2D(filters=512,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name = "conv-512b")(X)
    X = Conv2D(filters=512,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name = "conv-512c")(X)
    X = MaxPool2D(pool_size=(2,2),strides=(2,2), name="maxpool-5")(X)
    X = Dropout(0.2)(X)

    X = Flatten()(X)
    X = Dense(512, activation='relu', kernel_initializer='he_uniform', name = 'Dense-512-1')(X)  
    X= Dropout(0.5)(X)      
    output = Dense(num_classes, activation='softmax', name = 'output_layer')(X)

    model = Model(inputs=model.inputs, outputs=output)
    return model

############################################################################
# Use VGG19 with pre-trained weights, and apply transfer learning 

def VGG19_v1(input_shape = (224,224,3), classes = num_classes):

    model = VGG16(include_top=False, input_shape= input_shape, weights='imagenet')
    for layer in model.layers:
        layer.trainable = False  
    
    X = model.layers[-5].output
    X = Conv2D(filters=512,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name = "conv-512a")(X)
    X = Conv2D(filters=512,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name = "conv-512b")(X)
    X = Conv2D(filters=512,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name = "conv-512c")(X)
    X = Conv2D(filters=512,kernel_size=(3,3),padding="same", activation="relu", kernel_initializer = 'he_uniform', name = "conv-512d")(X)
    X = MaxPool2D(pool_size=(2,2),strides=(2,2), name="maxpool-5")(X)
    X = Dropout(0.2)(X)

    X = Flatten()(X)
    X = Dense(4096, activation='relu', kernel_initializer='he_uniform', name = 'Dense-4096-1')(X)  
    X = Dropout(0.5)(X)      
    X = Dense(4096, activation='relu', kernel_initializer='he_uniform', name = 'Dense-4096-2')(X)  
    X = Dropout(0.5)(X)  
    output = Dense(num_classes, activation='softmax', name = 'output_layer')(X)

    model = Model(inputs=model.inputs, outputs=output)
    return model

#model = VGG16_v1(input_shape = [img_width,img_height,3], classes = num_classes)
model = VGG19_v1(input_shape = [img_width,img_height,3], classes = num_classes)

model.summary()

############################################################################
# Use learnng rate schedule -> exponentialDecay

initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True)

#opt = SGD(lr=0.00002, momentum=0.9) 
opt = SGD(learning_rate = lr_schedule)
        
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
############################################################################ 
# Apply early stopping commands

monitor = EarlyStopping(monitor='val_loss', min_delta =0.001, patience = 15, verbose = 0,
                       mode = 'auto', restore_best_weights = True )
epochs = 250 
history = model.fit(train_ds, 
                    steps_per_epoch = len(train_ds),  #batch_size
                    epochs = epochs,
                    validation_data = val_ds,         #batch_size
                    validation_steps = len(val_ds),
                    callbacks=[monitor]
                    )

############################################################################
# Save a model

model.save("vgg19_t1.h5")

############################################################################
# Save history of training

df = pd.DataFrame(history.history)
filename = 'history_vgg19_t1.csv'
with open (filename, mode ='w') as f:
    df.to_csv(f)


