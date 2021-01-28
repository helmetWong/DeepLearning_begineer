# DeepLearning_2020Q
A basic introduction to learning CNN

#######################################################################################
# program: vgg16_intro.py

This program demonstrates how to train a Convolutional Neural Network CNN or ConvNet  
Firstly, we should download some photos as the dateset to train a VGG16 (a famous CNN model) 
We download and create a flower datasets with 5 classes of totaling 3,670 images.  
Secondly, we use the dataset to train a VGG16 model. 
Thirdly, after having trained the VGG16 model, we predict an arbitrary flowers image.  
The program will output the class name of the predicted image.  

input dataset for training:
Download the "flower_photos.tgz" from below website, and put into "date_dir"  
https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz  
data_dir = "E:/datasets/flower_photos/"  

input image for prediction: "673.jpg" or another images

output: print "roses" on the command prompt. 

#######################################################################################

***************************************************************************************

# program: vgg16_SaveModel.py".

This program continues from the program, "vgg16_SaveModel.py".
This program demonstrates how to load a CNN model and to predict a single image. 

output(1): Save a model
model.save("vgg16_t2.h5")

output(2): Save history of training
save histroy into 'history_vgg16_t2.csv'

***************************************************************************************

#######################################################################################

# program: vgg16_LoadModel.py".

This program continues from the program, "vgg16_SaveModel.py".
This program demonstrates how to load a CNN model and to predict a single image. 

input(1): vgg16_t2.h5

input(2): image for prediction: "673.jpg" or another images

output: imshow the input image, and prediction and accuracy in %

#######################################################################################

***************************************************************************************

# program: plot_learningCurve.py".

This program continues from the program, "vgg16_SaveModel.py".
This program demonstrates how to plot the training curves aftering training. 

input: history_vgg16_t2.csv

Output: two charts

***************************************************************************************

#######################################################################################

# program: vgg_Models.py".

This program continues from the program, "vgg16_intro.py" and "vgg16_SaveModel.py".  
This program demonstrates the architecture of vgg16 and vgg19  
There are two main methods to construct a CNN model.   
  Method A:  model = Sequential()      
  Method B:  model = Model(inputs=input, outputs=output)   

I prefer Method A for simple CNN.  I will use Method B for more complicated CNN.  
We also apply early stopping in training.  
We can then use "vgg16_LoadModel.py" to load the saved model and predict a single image. 

#######################################################################################

***************************************************************************************

# program vggTransferLearning.py".

This program continues from the program, "vgg16_intro.py" and "vgg16_SaveModel.py".  
This program demonstrates data augmentations and transfer learning.  
We use both vgg16 and vgg19.  

We also apply early stopping in training.    
We can then use "vgg16_LoadModel.py" to load the saved model and predict a single image. 

***************************************************************************************

