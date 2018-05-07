#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 16:25:00 2018

@author: ashutosh

#100DaysOfTensorflow

R1D11 - Save and Load TensorFlow Model

Source: http://stackabuse.com/tensorflow-save-and-restore-models/
"""
import tensorflow as tf  
import numpy as np  
import matplotlib.pyplot as plt  

import os
os.chdir("/Users/ashutosh/Documents/analytics/DeepLearning/TensorflowCodes/TFSaver")

# Model to estimate the horizontal and vertical shift in quadratic
# y = (x - h) ^ 2 + v  

# Clear the current graph in each run, to avoid variable duplication
tf.reset_default_graph()

# Create placeholders for X and Y
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# Initialize the two parameters that need to be learned
h_est = tf.Variable(0.0, name='hor_estimate')  
v_est = tf.Variable(0.0, name='ver_estimate')

#y_est will hold the estimated value
y_est = tf.square(x-h_est) + v_est

#Define a cost function : sq diff between y and y_est
cost = tf.square(y-y_est)

#Define optimizer

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)



# Use some values for the horizontal and vertical shift
h = 1  
v = -2

# Generate training data with noise
x_train = np.linspace(-2,4,201)  
noise = np.random.randn(*x_train.shape) * 0.4  
y_train = (x_train - h) ** 2 + v + noise

# Visualize the data 
plt.rcParams['figure.figsize'] = (10, 6)  
plt.scatter(x_train, y_train)  
plt.xlabel('x_train')  
plt.ylabel('y_train')  


"""
we define a Saver object and within the train_graph() method we go through 100 
iterations to minimize the cost function. The model is then saved to disk in 
each iteration, as well as after the optimization is finished. Each saving 
creates binary files on disk called "checkpoints".
"""

# Create a Saver object
saver = tf.train.Saver()



init = tf.global_variables_initializer()

# Run a session. Go through 100 iterations to minimize the cost
def train_graph():  
    with tf.Session() as sess:
        sess.run(init)
        for i in range(100):
            # Feed actual data to the train operation
            sess.run(optimizer, feed_dict={x: x_train, y: y_train})
            # Create a checkpoint in every iteration
            saver.save(sess, 'model_iter', global_step=i)

        # Save the final model
        saver.save(sess, 'model_final')
        h_ = sess.run(h_est)
        v_ = sess.run(v_est)
    return h_, v_

result = train_graph()  
print("h_est = %.2f, v_est = %.2f" % result) 


"""
".meta" files: containing the graph structure
".data" files: containing the values of variables
".index" files: identifying the checkpoint
"checkpoint" file: a protocol buffer with a list of recent checkpoints

Calling the tf.train.Saver() method, as shown above, would save all variables 
to a file. Saving a subset of your variables is possible by passing them as 
an argument through a list or a dict, 
for example: tf.train.Saver({'hor_estimate': h_est}).

A few other useful arguments of the Saver constructor, which enable control of 
the whole process, are:

max_to_keep: maximum number of checkpoints to keep,
keep_checkpoint_every_n_hours: a time interval for saving checkpoints
"""

## Restoring Models
tf.reset_default_graph()  
imported_meta = tf.train.import_meta_graph("model_final.meta")  

"""
The current graph could be explored using the following command 
tf.get_default_graph(). Now, the second step is to load the values of variables.

A reminder: values only exist within a session.
"""
with tf.Session() as sess:  
    imported_meta.restore(sess, tf.train.latest_checkpoint('./'))
    h_est2 = sess.run('hor_estimate:0')
    v_est2 = sess.run('ver_estimate:0')
    print("h_est: %.2f, v_est: %.2f" % (h_est2, v_est2))
    
plt.scatter(x_train, y_train, label='train data')  
plt.plot(x_train, (x_train - h_est2) ** 2 + v_est2, color='red', label='model')  
plt.xlabel('x_train')  
plt.ylabel('y_train')  
plt.legend()  