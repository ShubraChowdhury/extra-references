import numpy as np
from keras.utils import np_utils, plot_model
import tensorflow as tf
#tf.python.control_flow_ops = tf

# Set random seed
np.random.seed(42)

# Our data
X = np.array([[0,0],[0,1],[1,0],[1,1]]).astype('float32')
y = np.array([[0],[1],[1],[0]]).astype('float32')
#y = np_utils.to_categorical(y)
# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation

# Building the model
xor = Sequential()

# Add required layers
# xor.add()
xor.add(Dense(8, input_dim=2))
xor.add(Activation('tanh'))
xor.add(Dense(1))
xor.add(Activation('sigmoid'))

# Specify loss as "binary_crossentropy", optimizer as "adam",
# and add the accuracy metric
# xor.compile()
xor.compile(loss="binary_crossentropy", optimizer="adam", metrics = ["accuracy"])
#xor.compile(loss="categorical_crossentropy", optimizer="adam", metrics = ["accuracy"])

# Uncomment this line to print the model architecture
xor.summary()

# Fitting the model
history = xor.fit(X, y, nb_epoch=120, verbose=0)

# Scoring the model
score = xor.evaluate(X, y)
print("\nAccuracy: ", score[-1])

# Checking the predictions
print("\nPredictions:")
print(xor.predict_proba(X))


#plot_model(xor, to_file='model.png', show_shapes=False, show_layer_names=True, rankdir='TB')
