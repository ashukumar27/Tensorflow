# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 21:14:11 2018

@author: Ashutosh

#100Days of Code Challenge : R1D3 - Basic Linear Regression with Tensorflow
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



#Data
x = np.array([4.3,2.3,1.9,3.3,4.4,1.1,1.7,0.7,3,3.5,1.6,1.3,1.9,1.3])
y = np.array([18.5,13.3,11.2,17.7,18.6,8.1,9.6,6.4,15.3,15.1,7.9,9.7,10.7,9.4
])
plt.scatter(x,y)
plt.show()

#Parameters
LR = 0.01 # Learning Rate of the Gradient Descent OPtimizer
EPOCHS=1000 #Number of times going over the data
DISPLAY_BLOCK = 50 #To print intermittent results in epochs

#Define Placeholders
X = tf.placeholder(tf.float32) #placeholder for x defining the type
Y = tf.placeholder(tf.float32) #placeholder for y defining the type

#Define Variables

#Weight & Bias defined as variable since this value will change
#Also initialized with a random value
W = tf.Variable(np.random.randn(),tf.float32) 
b=tf.Variable(np.random.randn(),tf.float32) 

#Define the outcome
#y_ is the predicted values, cost will be calculated based on this and y
y_ = tf.add(tf.multiply(W,X),b)

#Define the cost
#cost function: mean of squared diff between actyal (y) and predicted (y_)
cost = tf.reduce_mean(tf.pow(y_-Y,2))

#Optimizer
##Gradient Descent Optimizer - Optimizes the 'variables' W and b to minimize 'cost'
optimizer = tf.train.GradientDescentOptimizer(LR).minimize(cost)

#Store the cost
costs = []

#Initialize the variables
#Mandatory initialization of variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(EPOCHS):
        sess.run(optimizer,feed_dict={X:x,Y:y})#Run optimizer for multiple iterations
        c = sess.run(cost,feed_dict={X:x,Y:y}) #Calculate cost
        costs.append(c)
        if epoch%DISPLAY_BLOCK==0:
            print("Weight: ",sess.run(W)," Bias: ",sess.run(b), "Cost :",c)
    print("Optimization Complete")
    
    #Plot cost
    plt.figure()
    plt.plot(costs)
    plt.show()

