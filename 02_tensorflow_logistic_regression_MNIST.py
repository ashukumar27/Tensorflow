#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 17:49:35 2017

@author: ashutosh

Logistic Regression implementation in Tensorflow
Classification of MNIST handwritten digits

original code:
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/logistic_regression.py
"""
from __future__ import print_function

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#IMport MNIST Data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#Parameters
learning_rate = 0.01
training_epochs = 25
batch_size=100
display_step = 1

## TF Graph Input
x = tf.placeholder(tf.float32, [None,784]) # Data shape of siz 28x28
y = tf.placeholder(tf.float32, [None,10]) # 10 Classes

#Set Model Weights
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#Construct model
pred = tf.nn.softmax(tf.matmul(x,W)+b) #Softmax

#Mininmizing error using cross-entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

#Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

## INitialize the variables and assign the cost 
init = tf.global_variables_initializer()

## Start training
with tf.Session() as sess:
    
    #Run the initializer
    sess.run(init)
    
    #Training Cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        #Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            #Run OPtimization OP (backprop) and cost op (to get loss value)
            _,c = sess.run([optimizer,cost], feed_dict = {x:batch_xs, y:batch_ys})
            
            #Compute average cost
            avg_cost += c/total_batch
        #Display cost per epoch step
        if (epoch+1)%display_step ==0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
        
    print("Optimization Finished")
    
    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

            