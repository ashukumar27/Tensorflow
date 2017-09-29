#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 15:06:00 2017

@author: ashutosh

TensorFlow Basics - Initial Codes

Serves no purpose other than a "Hello World!" program
"""

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


""" * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *"""
### Iris Data Classification - Basic Tensorflow Code

#Load DataSet
iris = tf.contrib.learn.datasets.load_dataset('iris')
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

##Build a 3 layer DNN with 10,20,10 units
feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(x_train)
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[10,20,10], n_classes=3)


#Fit and Predict
classifier.fit(x_train, y_train, steps=200)

predictions = list(classifier.predict(x_test, as_iterable=True))
score = accuracy_score(y_test, predictions)

score #1.0 - 100% Accuracy !!
""" * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *"""


### Very Basic Stuff
a = tf.add(3,5)
sess = tf.Session()
print (sess.run(a))
sess.close()

# or
a= tf.add(3,5)
with tf.Session() as sess:
    print(sess.run(a))

### Subgraphs
x=2
y=3
add_op = tf.add(x,y)
mul_op = tf.multiply(x,y)
pow_op = tf.pow(add_op, mul_op)

with tf.Session() as sess:
    print(sess.run(pow_op))


### Code 01
import tensorflow as tf

a=tf.constant(2)
b=tf.constant(3)
x = tf.add(a,b)
with tf.Session() as sess:
    ## Use tensorboard
    writer = tf.summary.FileWriter("/Users/ashutosh/Documents/analytics/DeepLearning/Learn/StanfordTensorflow/", sess.graph)
    print (sess.run(x))
writer.close()

## Run in CLI - run program, 
# Go to the output dir and run $tensorboard --logdir="./" --port 6006
# Open browser at http://localhost:6006 and go to Grahs in top tab

### Naming Constants

a = tf.constant(2, name="a")
b= tf.constant(3, name="b")
x = tf.add(a,b, name="add")

with tf.Session() as sess:
    writer = tf.summary.FileWriter("/Users/ashutosh/Documents/analytics/DeepLearning/Learn/StanfordTensorflow/", sess.graph)
    print(sess.run(x))
writer.close()


### Run Multiple
#tf.constant(value, dtype=None, shape=None,name='Const', verify_shape=False)
a = tf.constant([2,2], name='a')
b = tf.constant([[0,1], [2,3]], name = 'b')
x = tf.add(a,b,name="add")
y = tf.multiply(a,b,name="mul")

with tf.Session() as sess:
    x,y = sess.run([x,y])
    print (x,y)
    
### Zero values Tensor
#tf.zeros(shape, dtype=tf.float32, name=None)

x = tf.zeros([2,3], tf.int32)
#[[0 0 0],[0 0 0]]

### Tensors filled with a specific value
#tf.zeros_like(input_tensor, dtype=None, name=None, optimize=True)
#tf.ones_like(input_tensor, dtype=None, name=None, optimize=True)


#tf.fill(dims, value, name=None)
a = tf.fill([2,3],8)
with tf.Session() as sess:
    print(sess.run(a))
    
### Constants as Sequences
#tf.linspace(start, stop, num, name=None) # slightly different from np.linspace
a = tf.linspace(10.0,13.0, 4)

#tf.range(start, limit=None, delta=1, dtype=None, name='range')
a = tf.range(3,18,3)  #[3 6 9 12 15]

## Randomly Generated COnstants
#tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
#tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None,name=None)
#tf.random_uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None,name=None)
#tf.random_shuffle(value, seed=None, name=None)
#tf.random_crop(value, size, seed=None, name=None)
#tf.multinomial(logits, num_samples, seed=None, name=None)
#tf.random_gamma(shape, alpha, beta=None, dtype=tf.float32, seed=None, name=None)

### Random Seed
#tf.set_random_seed(seed)


""" * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *"""

## Operations on Matrices and Reshaping
a = tf.constant([3,6])
b = tf.constant([2,2])

tf.add(a,b) #>> [5,8]
tf.add_n(a,b,b) #>> Equivalent to a+b+b

tf.multiply(a,b) #Elementwise mult
tf.matmul(a,b) #error

tf.matmul(tf.reshape(a,[1,2]), tf.reshape(b,[2,1])) #dot product

tf.div(a,b) #[1,3]
tf.mod(a,b) #[1,0]


""" * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *"""
### TensorFlow Data Types
## 0-d Tensor or scaler
t_0=19
tf.zeros_like(t_0)#==> 0
tf.ones_like(t_0)#==> 1

##1-d Tensor or vector
t_1 = ['a','b','c']
tf.zeros_like(t_1) #=> ['','','']
tf.ones_like(t_1) #==> TypeError: Expected string, got 1 of type 'int' instead.

#2-d Tensor, or Matrix
t_2 = [
       [True, False, False],
       [False, False, True],
       [False, True, False]
       ]

tf.zeros_like(t_2)  #2x2 Tensor, All elements are False
tf.ones_like(t_2)   #2x2 Tensor, All elements are True


###  VARIABLES 

#create variable with a scaler value
a = tf.Variable(2, name="scaler")  #Capital V in Variable, small c in constant because Variable is a class
b = tf.Variable([2,3], name='vector') #Create variable b as a vector
c = tf.Variable([[0,1], [2,3]]) #Matrix
W = tf.Variable(tf.zeros([784,10])) #784x10 Tensor, filled with zeros

#### You have to initialize your variables

## All variables at once
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

## Initialize only a subset of variables
init_ab = tf.variables_initializer([a,b], name="init_ab")
with tf.Session() as sess:
    sess.run(init)
    
## Initialize onlu a single variable
W = tf.Variable(tf.zeros([784,10]))
with tf.Session() as sess:
    sess.run(W.initializer)
  
    
#### Eval() a variable

#W is a random 700x100 variable object
W = tf.Variable(tf.truncated_normal([700,10]))
with tf.Session() as sess:
    sess.run(W.initializer)
    print(W.eval())


## tf.Variable.assign()
W = tf.Variable(10)
W.assign(100)
with tf.Session() as sess:
    sess.run(W.initializer)
    print (W.eval()) #Not 100 but 10, since assign op has not been run in the session
    

#create a variable whose original value is 2
my_var = tf.Variable(2, name='my_var')

#assign a*2 to a and call that op a_times_2
my_var_times_2 = my_var.assign(2 * my_var)
with tf.Session() as sess:
    sess.run(my_var.initializer)
    sess.run(my_var_times_2)

# assign_add() and assign_sub()
my_var = tf.Variable(10)
with tf.Session() as sess:
    print(sess.run(my_var.initializer))
    print(sess.run(my_var.assign_add(10)))#increment by 10
    print(sess.run(my_var.assign_sub(8)))#decrement by 8
    



""" * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *"""
### Placeholders

#tf.placeholder(dtype, shape=None, name=None)

#a placeholder of type 32 bit float, shape is vector of 3 elements    
a = tf.placeholder(tf.float32,shape=[3])
#   contant - float32 - shape = vector of 3 
b = tf.constant([5,5,5], tf.float32)

#use the placeholder as that of a constnat of variable
c = a + b # Short for tf.add(a,b)
#this will give error while running since a does not have any value

### Feeding value to Placeholder via a dictionary
with tf.Session() as sess:
    #feed [1,2,3] to a via a dictionary
    print(sess.run(c, {a:[1,2,3]})) #the tensor 'a' is the key, not the string 'a'
    
###### Feeding values to  TF OPS
#create operations , tensors etc
a = tf.add(2,3)
b = tf.multiply(a,3)

with tf.Session() as sess:
    #define a dictionary that says to replace the values of 'a' with 15
    replace_dict = {a:15}
    #Run the session, passing in the new value
    print(sess.run(b, feed_dict = replace_dict))
    
""" * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *"""
### Lazy Loading
#Defer creating/initializing an object till it is needed

#Normal Loading
x= tf.Variable(10,name='x')
y = tf.Variable(20, name='y')
z = tf.add(x,y) #creating the node for add before executing the graph

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer =  tf.summary.FileWriter('/Users/ashutosh/Documents/analytics/DeepLearning/Learn/StanfordTensorflow/', sess.graph)
    for _ in range(10):
        sess.run(z)
    writer.close()

#Lazy Loading

x= tf.Variable(10,name='x')
y = tf.Variable(20, name='y')







