# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 22:34:16 2018

@author: Ashutosh

R1D6: Tensorflow  Functions

Workings of different tensorflow functions

Part I
1. Import 1-D, 2D, 3D, n-D Tensors
2. Matrix shape, size, rank, addition, subtraction, multiplication
3. Matrix INverse, Transpose

Part II
4. Sigmoid and Softmax Functions
5. comparing two vectors - tf.equal, tf.argmax
6. Reduce Mean, Reduce Sum

Part III
7. Reshape images to feed into CNN for tensorflow (channels last or first)
8. Conv2D Function - input and output
9. MaxPool2D
10. Dropout, Flatten and other functions

Part IV
11. Other Functions from tf.nn
12. Finish all functions from CNN 
"""

import tensorflow as tf

# Define Tensors


# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#          Part I
#          1. Import 1-D, 2D, 3D, n-D Tensors
#          2. Matrix shape, size, rank, addition, subtraction, multiplication
#          3. Matrix INverse, Transpose
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


# One Number
scaler = tf.constant(10,tf.int32)

# A vector of fixed length
vector1 = tf.random_normal([10,1], mean=-2, stddev=2,dtype="float32")
vector2 = tf.ones([10], tf.int32)
vector3 = tf.random_uniform([10])



# A matrix of 2 dimensions - not explicitely defined as a matrix
matrix = tf.random_uniform([2,3], dtype="float32")
matrix2 = tf.ones([3,3],dtype=tf.int32)


# A matrix of 2 dimensions -  explicitely defined as a matrix
matrix2 = tf.eye(2,3, dtype=tf.float32)

#Define a matrix with data

matrix4= tf.constant([1.0,2.0,3.0,4.0,5.0,6.0], dtype = tf.float32, shape=[1,2])

dt = tf.random_normal([3,28,28], dtype=tf.float32)
shp = tf.shape(dt)
rnk = tf.rank(dt)

with tf.Session() as sess:
    print(sess.run(rnk))
    
#Matrix Transpose
transpose = tf.matrix_transpose(matrix)

#Matrix Inverse
#Define a matrix - float32/float64
matrix3 = tf.random_normal([3,3],dtype=tf.float32)
inverse = tf.matrix_inverse(matrix3)
mul = tf.matmul(matrix3,inverse) # Output is an identity matrix

with tf.Session() as sess:
    print(sess.run(matrix4))
    print(sess.run(inverse))
    print(sess.run(mul))




# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#       Part II
#       4. Sigmoid and Softmax Functions
#       5. comparing two vectors - tf.equal, tf.argmax
#       6. Reduce Mean, Reduce Sum
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

# Function in Tensorflow

#Sigmoid
x = tf.nn.sigmoid(0.0)
print(x)

#Relu
x= tf.nn.relu(-10.0)

#Softmax
logits = tf.random_normal([2,3], dtype = tf.float32)
x=tf.nn.softmax(logits)

logits = tf.constant([10.0,5.0,20.0], dtype = tf.float32)
x=tf.nn.softmax(logits)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                        labels=tf_train_labels, logits=logits))

with tf.Session() as sess:
    print(sess.run(x))
    


# Comparing arrays: tf.equal, tf.argmax
x= tf.constant([1.0,20.0,30.0,40.0,5.0,6.0], dtype = tf.float32, shape=[2,3])
y= tf.constant([100.0,2.0,30.0,4.0,5.0,6.0], dtype = tf.float32, shape=[2,3])

a = tf.equal(x,y)
b = tf.argmax(x,axis=0) #Column Wise
c = tf.argmax(x,axis=1) # Row Wise

with tf.Session() as sess:
    print(sess.run(b))
    print(sess.run(c))
    
# Reduction : ReduceMean, ReduceSum

a =tf.reduce_sum(tf.reduce_mean(x,1))
b = tf.reduce_sum(x,1)

with tf.Session() as sess:
    print(sess.run(x))
    print(sess.run(b))


    




 