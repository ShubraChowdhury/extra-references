# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 16:13:58 2017

@author: shubra
"""

import tensorflow as tf

# Create TensorFlow object called tensor
hello_constant = tf.constant('Hello World!')

with tf.Session() as sess:
    # Run the tf.constant operation in the session
    output = sess.run(hello_constant)
    print(output)