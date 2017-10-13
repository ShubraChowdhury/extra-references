# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 08:53:28 2017

@author: shubra
"""

import tensorflow as tf

# The file path to save the data
save_file = './model.ckpt'

weights = tf.Variable(tf.truncated_normal([2,3]))

biases = tf.Variable(tf.truncated_normal([3]))

# Class used to save and/or restore Tensor Variables
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    print('Weight:')
    print(sess.run(weights))
    
    print('Bias:')
    print(sess.run(biases))
    
    saver.save(sess,save_file )

