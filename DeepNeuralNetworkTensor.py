# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 13:41:21 2017

@author: shubra
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

#mnist = input_data.read_data_sets('/datasets/ud730/mnist', one_hot=True, reshape=False)
mnist = input_data.read_data_sets(".", one_hot=True, reshape=False)

# Parameters
learning_rate = 0.001
training_epochs = 20
batch_size = 128  # Decrease batch size if you don't have enough memory
display_step = 1

n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)


"""
 Layer number of features, it determnes the size of hidden layer in neural network
 """
n_hidden_layer = 256 # layer number of features


# Store layers weight & bias
weights = {
    'hidden_layer': tf.Variable(tf.random_normal([n_input, n_hidden_layer])),
    'out': tf.Variable(tf.random_normal([n_hidden_layer, n_classes]))
}
biases = {
    'hidden_layer': tf.Variable(tf.random_normal([n_hidden_layer])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
        
# tf Graph input
x = tf.placeholder("float", [None, 28, 28, 1])
y = tf.placeholder("float", [None, n_classes])

"""
MNIST DB stores 28 by 28 pixel images with single chnnel. tf.reshape() the 28 by 28 matrices 
in x into row vector of 784 pixcels
"""
x_flat = tf.reshape(x, [-1, n_input])



# Hidden layer with RELU activation
layer_1 = tf.add(tf.matmul(x_flat, weights['hidden_layer']),biases['hidden_layer'])
layer_1 = tf.nn.relu(layer_1)
# Output layer with linear activation
logits = tf.add(tf.matmul(layer_1, weights['out']), biases['out'])

""" Calculate accuracy ADDED BY ME """
#correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
##print("accuracy ",accuracy)

"""
cost/loss and optimizer 
"""

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)\
    .minimize(cost)

init = tf.global_variables_initializer()

#print(tf.shape(features))
#
#print( tf.reshape(test_features, [-1, n_input]))
#test_features1 = tf.reshape(test_features, [-1, n_input])
#print(tf.shape(test_labels))
#print(tf.shape(test_features1))
#print(test_features[0])
#print(tf.reshape(test_features,[784]))
#test_features=tf.reshape(test_features[0],[1,784])

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
#            print(tf.shape(batch_y))
        # Display logs per epoch step
        if epoch % display_step == 0:
            c = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(c))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # Decrease test_size if you don't have enough memory
    test_size = 256
#    print("Accuracy:", accuracy.eval({x: mnist.test.images[:test_size], y: mnist.test.labels[:test_size]}))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
