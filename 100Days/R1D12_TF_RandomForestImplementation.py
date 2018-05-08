# -*- coding: utf-8 -*-
"""
Created on Tue May  8 11:09:21 2018

@author: Ashutosh

R1D12: #100DaysOfTensorflow

Implementing Random Forest in Tensorflow
"""
import os
os.chdir("D:\\DeepLearning\\datasets\\Iris")
import numpy as np
import pandas as pd
import tensorflow as tf
from __future__ import division
from __future__ import print_function

tf.logging.set_verbosity(tf.logging.DEBUG)


all_data = pd.read_csv("D:\\DeepLearning\\datasets\\Iris\\iris.csv")
all_data.head()

train = all_data[::2]
test = all_data[1::2]

x_train = train.drop(['species'], axis=1).astype(np.float32).values
y_train = train['species'].map({'setosa':0,'versicolor':1,'virginica':2}).astype(np.float32).values

#Define Params
params = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(
        num_classes=3, num_features=4, num_trees=2, max_nodes=10, split_after_samples=50)

print(params)

classifier = tf.contrib.tensor_forest.client.random_forest.TensorForestEstimator(params, model_dir="./")
classifier.fit(x=x_train, y=y_train)
classifier.evaluate(test)
















