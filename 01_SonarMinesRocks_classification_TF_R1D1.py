# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 13:38:05 2018

@author: Ashutosh

#100DaysOfTensorFlow : SonarMinesRocks Classification

R1D1
"""

datapath = "D:/DeepLearning/datasets/SonarMinesRocks/sonarminesrocks.csv"

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Read the dataset
def read_data():
    df =pd.read_csv(datapath)
    
    #define features and labels
    X = df[df.columns[0:60]].values
    y= df[df.columns[60]]
    
    #Encode the dependent variable
    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)
    Y = one_hot_encode(y_enc)
    print(X.shape)
    return(X,Y)

#Define Encoder Function
def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels),labels]=1
    return one_hot_encode

#Read the Dataset
X,Y = read_data()

#Shuffle the dataset to mix up the rows
X,Y = shuffle(X,Y, random_state = 7)

#Convert to train and test
X_train, X_test, y_train, y_test = train_test_split(X,Y, random_state=7, test_size=0.2)     
    

#Inspect shape
print("Train X: ",X_train.shape)
print("Test X: ",X_test.shape)    
print("Train Y: ",y_train.shape)
print("Test Y: ",y_test.shape)
    
#Define the parameters
learning_rate = 0.3
training_epochs = 1000
cost_history = np.empty(shape=[1], dtype=float)
n_dim=X.shape[1]
print("n_dim",n_dim)
n_class=2

model_path = "D:/DeepLearning/Tensorflow"

## Define the number of hidden layers and number of neurons for each layer
n_hidden_1 = 60
n_hidden_2 = 60
n_hidden_3 = 60
n_hidden_4 = 60

x = tf.placeholder(tf.float32,[None,n_dim])
W = tf.Variable(tf.zeros([n_dim,n_class]))
b = tf.Variable(tf.zeros([n_class]))
y_ = tf.placeholder(tf.float32,[None,n_class])


#Define the Model

def multilayer_perceptron(x,weights,biases):
    
    #Hidden layer with Sigmoid activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']),biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    
    #Hidden layer with sigmoid activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']),biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)
    
    #Hidden layer with sigmoid activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']),biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)
    
    #Hidden layer with sigmoid activation
    layer_4 = tf.add(tf.matmul(layer_3, weights['h2']),biases['b4'])
    layer_4 = tf.nn.relu(layer_4)
    
    #Output layer
    out_layer = tf.matmul(layer_4, weights['out'])+biases['out']
    return out_layer

#Define the weights and biases for each layer
weights = {
        'h1':tf.Variable(tf.truncated_normal([n_dim,n_hidden_1])),
        'h2':tf.Variable(tf.truncated_normal([n_hidden_1,n_hidden_2])),
        'h3':tf.Variable(tf.truncated_normal([n_hidden_2,n_hidden_3])),
        'h4':tf.Variable(tf.truncated_normal([n_hidden_3,n_hidden_4])),
        'out':tf.Variable(tf.truncated_normal([n_hidden_4,n_class])),
        }

biases = {
        'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
        'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
        'b3': tf.Variable(tf.truncated_normal([n_hidden_3])),
        'b4': tf.Variable(tf.truncated_normal([n_hidden_4])),
        'out': tf.Variable(tf.truncated_normal([n_class]))
        }


init = tf.global_variables_initializer()

saver = tf.train.Saver()

#Call you model defined

y = multilayer_perceptron(x,weights, biases)


### Define the cost function and optimizer

cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

### Calculate the cost and history of each epoch

mse_history = []
accuracy_history = []

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        sess.run(training_step, feed_dict = {x: X_train, y_:y_train})
        cost = sess.run(cost_function, feed_dict={x: X_train, y_:y_train})
        cost_history = np.append(cost_history,cost)
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        pred_y = sess.run(y, feed_dict={x: X_test})
        mse = tf.reduce_mean(tf.square(pred_y - y_test))    
        mse_ = sess.run(mse)
        mse_history.append(mse_)
        accuracy = (sess.run(accuracy, feed_dict = {x: X_train, y_:y_train} ))
        accuracy_history.append(accuracy)
        print('epoch: ',epoch,'-','cost: ',cost," - MSE: ", mse_," - Train Accuracy: ",accuracy)
        
    save_path  = saver.save(sess,model_path)
    print("Model saved in file %s" %save_path)
    
    #Plot MSE and Accuracy Graph
    
    plt.plot(mse_history,'r')
    plt.show()
    
    plt.plot(accuracy_history)
    plt.show()
 
    #Print the final accuracy
    correct_prediction  = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Test Accuracy:", (sess.run(accuracy, feed_dict={x: X_test, y_:y_test})))
    
    
    ## Print the final mean square error
    
    pred_y = sess.run(y, feed_dict = {x: X_test})
    mse = tf.reduce_mean(tf.square(pred_y - y_test))    
    print("MSE: $.4f" %sess.run(mse))
    








    
    