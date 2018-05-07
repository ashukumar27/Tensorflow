#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 16:05:28 2018

@author: ashutosh

R1D10  - Tensorboard

Vizualize graphs in TF using Tensorboard
"""

import tensorflow as tf
import matplotlib.pyplot as plt


a = tf.constant(3.0, tf.float32)
b = tf.constant(1.0, tf.float32)
c = tf.constant(5.0, tf.float32)

x = tf.multiply(tf.add(a,b),c)


with tf.Session() as sess:
    ## Use tensorboard
    writer = tf.summary.FileWriter("/Users/ashutosh/datasets/tensorboard/", sess.graph)
    print (sess.run(x))
writer.close()


### Running Tensorboard Graph
## Terminal : tensorboard --logdir="/Users/ashutosh/datasets/tensorboard/"
## Server Running on Browser: http://localhost:6006/