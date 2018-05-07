#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 16:01:59 2018

@author: ashutosh

#R1D9: Plotting Loss and Accuracy

"""

import tensorflow as tf
import matplotlib.pyplot as plt



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
    






