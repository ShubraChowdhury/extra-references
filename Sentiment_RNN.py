# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 23:02:35 2017

@author: shubra
"""

import numpy as np
import tensorflow as tf
print("Loaded ")

with open('C:/Training/udacity/AI_NanoDegree/Term2/11. Sentiment Prediction with RNN Videos/code/sentiment-network/reviews.txt', 'r') as f:
    reviews = f.read()
with open('C:/Training/udacity/AI_NanoDegree/Term2/11. Sentiment Prediction with RNN Videos/code/sentiment-network/labels.txt', 'r') as f:
    labels = f.read()
print("Done Reading")

print(reviews[:2000])
print(labels[:100])
#
from string import punctuation
all_text = ''.join([c for c in reviews if c not in punctuation])
print(all_text[:2000])
reviews = all_text.split('\n')

all_text = ' '.join(reviews)
words = all_text.split()

print(all_text[:2000])

# Create your dictionary that maps vocab words to integers here example the:336713,  thialnd:1 
from collections import Counter
counts = Counter(words)
#print((counts))
print("Length of Vocab ",len(counts))
import operator
max_value_of_counter=max(counts.values())


for k in counts.keys():
    if counts[k] == max_value_of_counter:
        print(k)      
kv = next(( (k , max_value_of_counter)  for k in counts.keys()  if counts[k] == max_value_of_counter ),None)        
print('Max occuring word is: "{}" --and {} times it has repeated'.format(kv[0],kv[1]))

print(max_value_of_counter)
#print(counts.values().index['336713'])
print("Number of occurance of word  'THE' and 'AND' is {},{}" .format(counts['the'],counts['and']))
print("Number of occurance of word  thialnd is {}" .format(counts['thialnd']))

# Sort By position
vocab = sorted(counts, key=counts.get, reverse=True)
#print(vocab[0], len(vocab))
#vocab= set(words)
vocab_to_int = {c:i for i , c in enumerate(vocab,1)}

print("Position of Thiland",vocab_to_int['thialnd'],"Position of The",vocab_to_int['the'],"Length of Vocab ",len(vocab_to_int))

# Convert the reviews to integers, same shape as reviews list, but with integers
#reviews_ints = (c:i for i , c in enumerate(reviews))

#print(reviews.split()) 'list' object has no attribute 'split'

#rv =[]
#for each in reviews:
#    rv.append(vocab_to_int[each.split()])
    
reviews_ints = []
for each in reviews:
    reviews_ints.append([vocab_to_int[word] for word in each.split()])
    
print(len(reviews_ints), reviews_ints[:2])



labels = labels.split('\n')
#print(labels)

labels =np.array([1 if each == 'positive' else 0 for each in labels])
print(labels)

from collections import Counter
review_lens = Counter([len(x) for x in reviews_ints])
print("Zero-length reviews: {}".format(review_lens[0]))
print("Maximum review length: {}".format(max(review_lens)))

# Filter out that review with 0 length
zero_idx = [idx_zero for idx_zero,zero_val in enumerate(reviews_ints) if len(zero_val) == 0]
print("Number of Reviews with Zero Length :",len(zero_idx))
non_zero_idx = [idx_nz for idx_nz,nz_val in enumerate(reviews_ints) if len(nz_val) != 0]
print("Number of Reviews with Non Zero Length :",len(non_zero_idx))


for i in zero_idx:
    print(i , 'has length zero ', reviews_ints[i])
    
reviews_ints = [reviews_ints[i] for i in non_zero_idx ]
labels = np.array([labels[i] for i in non_zero_idx])
print(len(reviews_ints),labels.shape)


print(type(reviews_ints))
print(reviews_ints[0])

seq_len = 200
seq_len1 = 50
features = np.zeros((len(reviews_ints),seq_len),dtype=int)
print(features[:3,:4])
features1 = np.zeros((len(reviews_ints),seq_len1),dtype=int)
print(features1[:3,:4])
for i, row in enumerate(reviews_ints):
    #print("i =", i, row)
    #print( np.array(row))
    features[i, -len(row):] = np.array(row)[:seq_len]
    features1[i, -len(row):] = np.array(row)[:seq_len1]


print(features[:2,60:70])   
print(features1[:2,:10])



split_frac = 0.8
train_idx = int(len(features)*split_frac)

train_x, val_x = features[:train_idx], features[train_idx:]
train_y, val_y = labels[:train_idx], labels[train_idx:]

#split in half to create the validation and testing data
val_test_idx= int(len(val_x)*0.5)
val_x, test_x = val_x[val_test_idx:], val_x[:val_test_idx]
val_y, test_y = val_y[val_test_idx:], val_y[:val_test_idx]

print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape), 
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))



lstm_size = 256
lstm_layers = 1
batch_size = 500
learning_rate = 0.001

n_words = len(vocab_to_int) + 1 # Adding 1 because we use 0's for padding, dictionary started at 1
print("n_words :",n_words)
# Create the graph object
graph = tf.Graph()
# Add nodes to the graph
with graph.as_default():
    inputs_ = tf.placeholder(tf.int32,[None,None], name='inputs')
    labels_ = tf.placeholder(tf.int32,[None,None], name ='labels')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
print(inputs_.shape,labels_.shape,keep_prob.shape,type(keep_prob))


# Size of the embedding vectors (number of units in the embedding layer)
embed_size = 300 

with graph.as_default():
    embedding = tf.Variable(tf.random_uniform((n_words,embed_size),-1,1))
    embed = tf.nn.embedding_lookup(embedding,inputs_)
print(type(embedding),embed.shape)
print(embed)
print(embedding)


with graph.as_default():
    # Your basic LSTM cell
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    
    # Add dropout to the cell
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    
    # Stack up multiple LSTM layers, for deep learning
    cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)
    
    # Getting an initial state of all zeros
    initial_state = cell.zero_state(batch_size, tf.float32)
    
    
with graph.as_default():
    outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=initial_state)



with graph.as_default():
    predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=tf.sigmoid)
    cost = tf.losses.mean_squared_error(labels_, predictions)
    
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


with graph.as_default():
    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


def get_batches(x, y, batch_size=100):
    
    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]    
        
        
epochs = 10

with graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 1
    for e in range(epochs):
        state = sess.run(initial_state)
        
        for ii, (x, y) in enumerate(get_batches(train_x, train_y, batch_size), 1):
            feed = {inputs_: x,
                    labels_: y[:, None],
                    keep_prob: 0.5,
                    initial_state: state}
            loss, state, _ = sess.run([cost, final_state, optimizer], feed_dict=feed)
            
            if iteration%5==0:
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Train loss: {:.3f}".format(loss))

            if iteration%25==0:
                val_acc = []
                val_state = sess.run(cell.zero_state(batch_size, tf.float32))
                for x, y in get_batches(val_x, val_y, batch_size):
                    feed = {inputs_: x,
                            labels_: y[:, None],
                            keep_prob: 1,
                            initial_state: val_state}
                    batch_acc, val_state = sess.run([accuracy, final_state], feed_dict=feed)
                    val_acc.append(batch_acc)
                print("Val acc: {:.3f}".format(np.mean(val_acc)))
            iteration +=1
    saver.save(sess, "C:/Training/udacity/AI_NanoDegree/Term2/11. Sentiment Prediction with RNN Videos/code/sentiment-network/checkpoints/sentiment.ckpt")        
    
    
test_acc = []
with tf.Session(graph=graph) as sess:
    saver.restore(sess, tf.train.latest_checkpoint('C:/Training/udacity/AI_NanoDegree/Term2/11. Sentiment Prediction with RNN Videos/code/sentiment-network/checkpoints'))
    test_state = sess.run(cell.zero_state(batch_size, tf.float32))
    for ii, (x, y) in enumerate(get_batches(test_x, test_y, batch_size), 1):
        feed = {inputs_: x,
                labels_: y[:, None],
                keep_prob: 1,
                initial_state: test_state}
        batch_acc, test_state = sess.run([accuracy, final_state], feed_dict=feed)
        test_acc.append(batch_acc)
    print("Test accuracy: {:.3f}".format(np.mean(test_acc)))    
    
    
    