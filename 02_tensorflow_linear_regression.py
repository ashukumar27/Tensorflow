#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 17:06:08 2017

@author: ashutosh

Linear Regression with simple data using Tensorflow

This code is intended to show the workings of a Tensorflow model rather than 
complex analysis or regularization
"""

#Import libraries
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
rng = np.random

#Parameters
learning_rate = 0.01
training_epochs = 1000
display_step=50

#Training Data
train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]


### TF Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

#Set Model weights
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")


####  Model Building   #####

#Construct a linear Model
pred = tf.add(tf.multiply(X,W),b)

#Mean Squared Error
cost = tf.reduce_sum(tf.pow(pred-Y,2))/(2*n_samples)

## Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


#Intitalize the variables
init = tf.global_variables_initializer()

#Start Training
with tf.Session() as sess:
    
    #Run the initializer
    sess.run(init)
    
    #fit all the training data
    for epoch in range(training_epochs):
        for (x,y) in zip(train_X,train_Y):
            sess.run(optimizer,feed_dict={X:x,Y:y})
    
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

            
    