#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 12:26:01 2018

@author: ashutosh


#100DaysOfTensorFlow : Logistic Regression Classification using Tensorflow

R1D4
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf


#IMport MNIST Data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#Parameters for the model
learning_rate = 0.01
epochs = 25
batch_size=100
display_step=1

## Build TF Graph
x = tf.placeholder(tf.float32, [None,784]) #28x28 images
y= tf.placeholder(tf.float32,[None,10]) #10 Classes

## Define Model weighhts
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

## Build Model
pred = tf.nn.softmax(tf.matmul(x,W)+b)


## Cost Function

cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

#OPtimizer - Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#Initialize the variables
init = tf.global_variables_initializer()

#Run Session
with tf.Session() as sess:
    
    #Run Initializer
    sess.run(init)
    
    #Training Cycle
    for epoch in range(epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        
        #Loop Over all the batches
        for i in range(total_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict = {x:batch_xs, y: batch_ys})
            c = sess.run(cost, feed_dict = {x:batch_xs, y: batch_ys})
            
            avg_cost = c/total_batch
        #Display cost per epoch step
        if((epoch+1)%display_step==0):
            print("Epoch: ",epoch+1," Cost: ",avg_cost)
        
        #Correct Prediction

    correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    
    print("Accuracy : ",accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))            
                
            
        
        
        
        
        
        
        
        
        