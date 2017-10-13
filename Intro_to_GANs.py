# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 10:37:18 2017

@author: shubra
"""

import pickle as pkl
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist= input_data.read_data_sets("MNIST_data")

"""
Model Input
"""
def model_inputs(real_dim,z_dim):
    inputs_real = tf.placeholder(tf.float32,(None,real_dim),name ='inputs_real')
    inputs_z = tf.placeholder(tf.float32,(None,z_dim),name ='inputs_z')
    return inputs_real,inputs_z
"""
GENERATOR FUNCTION
"""
def generator(z,out_dim,n_units=128,reuse=False,alpha=0.01):
    
    with tf.variable_scope('generator',reuse=reuse):
        """ HIDDEn LAYER """
        h1=tf.layers.dense(z,n_units,activation=None)
        """ Leaky ReLU"""
        h1= tf.maximum(h1*alpha,h1)
        """ Logits & Out """
        logits=tf.layers.dense(h1,out_dim,activation=None)
        out = tf.tanh(logits)
        return out
"""
DISCRIMINATOR
"""
def discriminator(x,n_units=128,reuse=False,alpha=0.01):
    with tf.variable_scope('discriminator',reuse=reuse):
        """HIDDEN """
        h1=tf.layers.dense(x,n_units,activation=None)
        """ReLU """
        h1= tf.maximum(h1*alpha,h1)
        """Logits and out """
        logits = tf.layers.dense(h1,1,activation=None)
        out = tf.sigmoid(logits)
        return out,logits
    
""" HYPER PARAMETERS """
"""  Size 28*28 for MNIST Image  """
input_size=784 
""" SIZE OF LATENT Vector  Generator """
z_size=100 
""" Size of Generator and Discriminator """
g_hidden_size=128
d_hidden_size=128
""" FACTOR For ReLU """
alpha=0.01
""" Label Smoothing """
smooth=0.1


tf.reset_default_graph()
""" CREATE Input Place Holder """
input_real,input_z = model_inputs(input_size,z_size)
""" GENERATOR Network """
g_model = generator(input_z,input_size,n_units=g_hidden_size,alpha=alpha)

""" DISCRIMANATOR """
d_model_real,d_logits_real = discriminator(input_real,n_units=d_hidden_size,alpha=alpha)
d_model_fake,d_logits_fake = discriminator(g_model,reuse=True,n_units=d_hidden_size,alpha=alpha)

"""
Discriminator and Generator Losses

Now we need to calculate the losses, which is a little tricky. For the discriminator, the total loss is the sum of the losses for real and fake images, d_loss = d_loss_real + d_loss_fake. The losses will by sigmoid cross-entropies, which we can get with tf.nn.sigmoid_cross_entropy_with_logits. We'll also wrap that in tf.reduce_mean to get the mean for all the images in the batch. So the losses will look something like 
tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))


For the real image logits, we'll use d_logits_real which we got from the discriminator in the cell above. For the labels, we want them to be all ones, since these are all real images. To help the discriminator generalize better, the labels are reduced a bit from 1.0 to 0.9, for example, using the parameter smooth. This is known as label smoothing, typically used with classifiers to improve performance. In TensorFlow, it looks something like labels = tf.ones_like(tensor) * (1 - smooth)

The discriminator loss for the fake data is similar. The logits are d_logits_fake, which we got from passing the generator output to the discriminator. These fake logits are used with labels of all zeros. Remember that we want the discriminator to output 1 for real images and 0 for fake images, so we need to set up the losses to reflect that.

Finally, the generator losses are using d_logits_fake, the fake image logits. But, now the labels are all ones. The generator is trying to fool the discriminator, so it wants to discriminator to output ones for fake images.

"""

# Calculate losses
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=
                                                                    tf.ones_like(d_logits_real) * (1 - smooth)))

d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                     labels=tf.zeros_like(d_logits_real)))

d_loss = d_loss_real+d_loss_fake

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, 
                                                                labels=tf.ones_like(d_logits_fake)))


# Optimizers
learning_rate = 0.002

# Get the trainable_variables, split into G and D parts
t_vars = tf.trainable_variables() 
g_vars = [var for var in t_vars if var.name.startswith('generator')]
d_vars = [var for var in t_vars if var.name.startswith('discriminator')]

d_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)
g_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)


# Optimizers
learning_rate = 0.002

# Get the trainable_variables, split into G and D parts
t_vars = tf.trainable_variables() 
g_vars = [var for var in t_vars if var.name.startswith('generator')]
d_vars = [var for var in t_vars if var.name.startswith('discriminator')]

d_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)
g_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)


batch_size = 100
epochs = 100
samples = []
losses = []
saver = tf.train.Saver(var_list = g_vars)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        for ii in range(mnist.train.num_examples//batch_size):
            batch = mnist.train.next_batch(batch_size)
            
            # Get images, reshape and rescale to pass to D
            batch_images = batch[0].reshape((batch_size, 784))
            batch_images = batch_images*2 - 1
            
            # Sample random noise for G
            batch_z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            
            # Run optimizers
            _ = sess.run(d_train_opt, feed_dict={input_real: batch_images, input_z: batch_z})
            _ = sess.run(g_train_opt, feed_dict={input_z: batch_z})
        
        # At the end of each epoch, get the losses and print them out
        train_loss_d = sess.run(d_loss, {input_z: batch_z, input_real: batch_images})
        train_loss_g = g_loss.eval({input_z: batch_z})
            
        print("Epoch {}/{}...".format(e+1, epochs),
              "Discriminator Loss: {:.4f}...".format(train_loss_d),
              "Generator Loss: {:.4f}".format(train_loss_g))    
        # Save losses to view after training
        losses.append((train_loss_d, train_loss_g))
        
        # Sample from generator as we're training for viewing afterwards
        sample_z = np.random.uniform(-1, 1, size=(16, z_size))
        gen_samples = sess.run(
                       generator(input_z, input_size, n_units=g_hidden_size, reuse=True, alpha=alpha),
                       feed_dict={input_z: sample_z})
        samples.append(gen_samples)
        saver.save(sess, './checkpoints/generator.ckpt')

# Save training generator samples
with open('train_samples.pkl', 'wb') as f:
    pkl.dump(samples, f)
    
fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator')
plt.plot(losses.T[1], label='Generator')
plt.title("Training Losses")
plt.legend()

def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(7,7), nrows=4, ncols=4, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((28,28)), cmap='Greys_r')
    
    return fig, axes


# Load samples from generator taken while training
with open('train_samples.pkl', 'rb') as f:
    samples = pkl.load(f)
    
_ = view_samples(-1, samples)


rows, cols = 10, 6
fig, axes = plt.subplots(figsize=(7,12), nrows=rows, ncols=cols, sharex=True, sharey=True)

for sample, ax_row in zip(samples[::int(len(samples)/rows)], axes):
    for img, ax in zip(sample[::int(len(sample)/cols)], ax_row):
        ax.imshow(img.reshape((28,28)), cmap='Greys_r')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        
        
saver = tf.train.Saver(var_list=g_vars)
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    sample_z = np.random.uniform(-1, 1, size=(16, z_size))
    gen_samples = sess.run(
                   generator(input_z, input_size, n_units=g_hidden_size, reuse=True, alpha=alpha),
                   feed_dict={input_z: sample_z})
_ = view_samples(0, [gen_samples])




        