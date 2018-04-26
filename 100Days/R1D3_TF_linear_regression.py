#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 14:57:15 2018

@author: ashutosh

#100DaysOfTensorFlow : Linear Regression of Boston Housing Dataset

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
rng = np.random
# Read data in csv

data = pd.read_csv("/Users/ashutosh/datasets/BostonHousing/boston.csv")
data.head()

## Read alternate data
#Training Data
train_X = pd.DataFrame([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = pd.DataFrame([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])


plt.scatter(train_X,train_Y)
plt.show()

## Parameters of Tensorflow
#Parameters
learning_rate = 0.01
training_epochs = 1000
display_step=20

n_samples =train_X.shape[0]

### Build the Graph
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

## Variables
W = tf.Variable(np.random.randn(), name="Weight")
b = tf.Variable(np.random.randn(), name="Bias")

#### Build the model
pred = tf.add(tf.multiply(W,X),b)


#cost
cost = tf.reduce_sum(tf.pow(pred-Y,2))/(2*n_samples)


#Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

## Intitialize the variables 
init = tf.global_variables_initializer()

#Start Training
with tf.Session() as sess:
    
    #Run the initializer
    sess.run(init)
    
    #fit all the training data
    for epoch in range(training_epochs):
        sess.run(optimizer,feed_dict={X:train_X,Y:train_Y})
        #Display logs for each epoch step
        if (epoch+1)%display_step==0:
            c=sess.run(cost,feed_dict={X:train_X,Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))
    
    print("Optimizer Finished")
    training_cost = sess.run(cost,feed_dict={X:train_X,Y:train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')
    
    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()

            
            

