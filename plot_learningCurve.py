import numpy as np
import h5py
import pandas as pd
import os
import matplotlib.pyplot as plt

################################################################################################
#
# This program continues from the program, "vgg16_SaveModel.py".
# This program demonstrates how to plot the training curves aftering training. 
#
################################################################################################

dir = "./history/"
filename = "history_vgg16_t2.csv"

trainingTime = 15

df = pd.read_csv(os.path.join(dir, filename))

x1 = df['accuracy']
x2 = df['val_accuracy']
x3 = df['loss']
x4 = df['val_loss']

def plot_learningCurve(x1, x2, x3, x4, epoch):
    #fig, axes = plt.subplots (1, 2)
    plt.figure(1)
    epoch_range = range(1,epoch+1)
    plt.plot(epoch_range, x1)
    plt.plot(epoch_range, x2)
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train- accuracy', 'Val-accuracy'], loc = 'upper left')
    #plt.show()

    plt.figure(2)
    plt.plot(epoch_range, x3)
    plt.plot(epoch_range, x4)
    plt.title ('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train-loss', 'Val-loss'], loc = 'upper right')

    plt.tight_layout()
    plt.show()

plot_learningCurve(x1, x2, x3, x4, trainingTime)
