# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 22:57:14 2017

@author: shubra
"""
import tensorflow as tf

save_file = 'C:/Training/udacity/AI_NanoDegree/Term2/5. Tensorflow/train_model.ckpt'
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    saver.restore(sess, save_file)

    test_accuracy = sess.run(
        accuracy,
        feed_dict={features: mnist.test.images, labels: mnist.test.labels})

print('Test Accuracy: {}'.format(test_accuracy))