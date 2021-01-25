# DeepLearning_Jan2020
A basic introduction to learn CNN


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

# program: vgg16_intro.py

# input dataset for training:
# Download the "flower_photos.tgz" from below website, and put into "date_dir"
# /datasets/flowers_photos/daisy/
# /datasets/flowers_photos/dandelion/
# /datasets/flowers_photos/roses/
# /datasets/flowers_photos/sunflowers/
# /datasets/flowers_photos/tulips/
# https://www.tensorflow.org/tutorials/images/classification
# https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
# data_dir = "E:/datasets/flower_photos/"

# input image for prediction:
# "608.jpg" or another images

# output:
# print "roses" on the command line. 
