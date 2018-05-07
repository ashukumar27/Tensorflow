#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 11:52:41 2018

@author: ashutosh

#100DaysOfTensorflow

R1D11 - Save and Load Keras Model

"""

## Saving and loading model with KERAS

from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

import numpy as np
import pandas as pd
import os

os.chdir("/Users/ashutosh/datasets/PimaIndianDiabetes/")

np.random.seed(42)

#Load Data - Pima Indian Diabetes Dataset
data = pd.read_csv("/Users/ashutosh/datasets/PimaIndianDiabetes/data.csv", header=None)
data.head()

X = data.values[:,0:8]
Y = data.values[:,8]

#Create Model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu', kernel_initializer='glorot_uniform'))
model.add(Dense(8, activation='relu', kernel_initializer='glorot_uniform'))
model.add(Dense(1, activation='sigmoid', kernel_initializer='uniform'))

#Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Fit
model.fit(X,Y, epochs=150, batch_size=10, verbose=2)

# evaluate the model
scores = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
 
#Serialize model to JSON
model_json = model.to_json()
with open("model.json","w") as json_file:
    json_file.write(model_json)

#Serialize weights to HDF5
model.save_weights("model.h5")
print("Saved to Disk")



### Load JSON File and create model


json_file = open("model.json","r")
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)

## Load Weights into the new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

##Evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X,Y, verbose=1)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))




