# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 18:55:26 2017

@author: shubra
"""

# Solution is available in the other "solution.py" tab
import tensorflow as tf

softmax_data = [0.7, 0.2, 0.1]
one_hot_data = [1.0, 0.0, 0.0]

softmax = tf.placeholder(tf.float32)
one_hot = tf.placeholder(tf.float32)

# TODO: Print cross entropy from session
cross_entropy = -tf.reduce_sum(tf.multiply(one_hot,tf.log(softmax)))
with tf.Session() as sess:
    print(sess.run(cross_entropy, feed_dict={one_hot : one_hot_data ,softmax: softmax_data}))
