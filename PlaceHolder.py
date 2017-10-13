# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 16:16:16 2017

@author: shubra
"""

import tensorflow as tf


def run():
    output = None
    x = tf.placeholder(tf.int32)

    with tf.Session() as sess:
        # TODO: Feed the x tensor 123
        #x =123
        output = sess.run(x,feed_dict={x: 123})
    print(output)
    return output

run()