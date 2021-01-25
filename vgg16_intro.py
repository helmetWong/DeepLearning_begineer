import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model 
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

################################################################################################
#
# This program demonstrates how to train a Convolutional Neural Network CNN or ConvNet -> VGG16
# Firstly, we should download some photos as the dateset to train a VGG16
# We download and create a flower datasets with 5 classes of totaling 3,670 images
# Secondly, we uses the dataset to train a VGG16 model,which it is directly downloaded from keras
# Thirdly, after having trained the VGG16 model, we predict an arbitrary flowers image. 
# The program will output the class name of the predicted image. 
#
################################################################################################


batch_size = 128    # Using Quadro RTX 5000 with 16G video RAM, "192" and "256" are too large.
                    # Reduce the batch_size if the vidoe RAM is not enough
img_height = 224    # (224, 224, 3) standard input size for VGG model
img_width = 224

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

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


class_names = train_ds.class_names
print(class_names)              # ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
print (len(class_names))        # num_classes = 5
num_classes = len(class_names)  


############################################################################
# plot / display the input date if you wish
#import matplotlib.pyplot as plt

#plt.figure(figsize=(10, 10))
#for images, labels in train_ds.take(1):
#  for i in range(9):
#    ax = plt.subplot(3, 3, i + 1)
#    plt.imshow(images[i].numpy().astype("uint8"))
#    plt.title(class_names[labels[i]])
#    plt.axis("off")
#plt.show()

############################################################################
# normailize the pixels in input images
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

############################################################################
# Use VGG16 without pre-trained weights
model = VGG16(weights = None, classes = num_classes)
model.summary()
opt = Adam(lr=0.001)

model.compile(optimizer=opt,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs = 10      # epochs = 20 -> accuracy = 0.7279  # epochs = 30 -> accuracy = 0.9916
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

############################################################################
# Find any flower image for prediciton
img = image.load_img("608.jpg",target_size=(224,224))
img = np.asarray(img)
img = np.expand_dims(img, axis=0)
#from keras.models import load_model
#saved_model = load_model("vgg16_1.h5")

output = model.predict(img)
index_max = np.argmax(output)
print("*"*120)
print()
print(class_names[index_max])

