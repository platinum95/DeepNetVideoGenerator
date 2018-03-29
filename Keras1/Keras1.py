# Import the necessary modules

from DataHandler import DataHandler
#import keras
#from keras import datasets
#from keras.layers import Dense, Flatten, Dropout, Activation
#from keras.layers import PReLU, LeakyReLU, Conv2D, MaxPool2D, Lambda
#from keras.regularizers import l2

#from keras.models import model_from_json

#from IPython.display import clear_output

#import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib.ticker import MaxNLocator

#import pickle
#import sklearn as skl

#from sklearn import datasets, linear_model
#from sklearn.model_selection import cross_val_score


## Define some useful functions
#class PlotLossAccuracy(keras.callbacks.Callback):
#    def on_train_begin(self, logs={}):
#        self.i = 0
#        self.x = []
#        self.acc = []
#        self.losses = []
#        self.val_losses = []
#        self.val_acc = []
#        self.logs = []

#    def on_epoch_end(self, epoch, logs={}):
        
#        self.logs.append(logs)
#        self.x.append(int(self.i))
#        self.losses.append(logs.get('loss'))
#        self.val_losses.append(logs.get('val_loss'))
#        self.acc.append(logs.get('acc'))
#        self.val_acc.append(logs.get('val_acc'))
        
#        self.i += 1
        
#        clear_output(wait=True)
#        plt.figure(figsize=(16, 6))
#        plt.plot([1, 2])
#        plt.subplot(121) 
#        plt.plot(self.x, self.losses, label="train loss")
#        plt.plot(self.x, self.val_losses, label="validation loss")
#        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
#        plt.ylabel('loss')
#        plt.xlabel('epoch')
#        plt.title('Model Loss')
#        plt.legend()
#        plt.subplot(122)         
#        plt.plot(self.x, self.acc, label="training accuracy")
#        plt.plot(self.x, self.val_acc, label="validation accuracy")
#        plt.legend()
#        plt.ylabel('accuracy')
#        plt.xlabel('epoch')
#        plt.title('Model Accuracy')
#        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
#        plt.show();
        
#def save_model_to_disk():    
#    # save model and weights (don't change the filenames)
#    model_json = model.to_json()
#    with open("model.json", "w") as json_file:
#        json_file.write(model_json)
#    # serialize weights to HDF5
#    model.save_weights("model.h5")
#    print("Saved model to model.json and weights to model.h5")


dataHandler = DataHandler()

dataHandler.getNextData()
