import numpy as np
import h5py
import pandas as pd
import tensorflow as tf
import os
import cv2 as cv
from tensorflow.keras.preprocessing import image
#from keras.preprocessing.image import load_img
#from keras.preprocessing.image import img_to_array

################################################################################################
#
# This program continues from the program, "vgg16_SaveModel.py".
# This program demonstrates how to load a CNN model and to predict a single image. 
#
################################################################################################

loaded_model = "./weights/vgg16_t2.h5"
model = tf.keras.models.load_model(loaded_model)
model.summary()

## Load and predict one image

#test_data_dir = "E:/datasets/flower_photos/roses/"
#filename = "295257304_de893fc94d.jpg"
test_data_dir = ""
filename = "673.jpg"

test_image = image.load_img(os.path.join(test_data_dir, filename), target_size =(224,224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
prediction = model.predict(test_image)

class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
index_max = np.argmax(prediction)

pred = class_names[index_max]
preList = np.reshape(prediction, -1)
pre = preList.tolist()
print (pre)
precentage = str("{:.1f}%".format(pre[index_max] * 100))
#print (precentage)

img = cv.imread(os.path.join(test_data_dir, filename), -1)

cv.imshow(pred + " (" + precentage +  ")", img)
cv.waitKey()