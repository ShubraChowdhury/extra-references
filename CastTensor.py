# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 16:19:03 2017

@author: shubra
"""

import tensorflow as tf

# TODO: Convert the following to TensorFlow:
x = 10
y = 2
z = x/y - 1

x = tf.constant(x)
y= tf.constant(y)
z = tf.subtract(tf.divide(x,y),tf.cast(tf.constant(1),tf.float64))

# TODO: Print z from a session
with tf.Session() as sess:
    
    output = sess.run(z)    
    print(output)