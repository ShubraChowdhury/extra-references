# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 18:31:40 2017

@author: shubra
"""

import tensorflow as tf

logit_data = [2.0,1.0,0.1]

logits = tf.placeholder(tf.float32)

softmax = tf.nn.softmax(logits)

with tf.Session() as sess:
    output = sess.run(softmax , feed_dict={logits: logit_data})
    
print(output)