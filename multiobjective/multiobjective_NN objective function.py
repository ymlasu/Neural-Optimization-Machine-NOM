import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Concatenate, add
from tensorflow.keras import Input
from tensorflow.keras import optimizers
from keras import backend as K
import os

xx_train1 = np.linspace(0, 10, 100)
xx_train2 = np.linspace(5, 15, 100)
xv, yv = np.meshgrid(xx_train1, xx_train2)
x_train1 = xv.flatten()
x_train2 = yv.flatten()
y_train1 = 1/10 *((x_train1 - 3)**2 + (x_train2 -7)**2)
y_train2 = 1/10 *((x_train1 - 9)**2 + (x_train2 -8)**2)


def loss_function(output, y):
    return tf.reduce_mean(tf.math.square(output - y))


# Model 1
epochs = 500
batch_size = 10
learning_rate = 0.001
InputLayer1 = Input(shape=(1,))
InputLayer2 = Input(shape=(1,))
Layer_2 = Dense(20,activation="tanh")(Concatenate(axis=1)([InputLayer1, InputLayer2]))
OutputLayer = Dense(1, activation="linear")(Layer_2)

y_real = Input(shape=(1,))

lossF = loss_function(OutputLayer, y_real)
model1 = Model(inputs=[InputLayer1, InputLayer2, y_real], outputs=OutputLayer)
model1.add_loss(lossF)
adamOptimizer = optimizers.Adam(learning_rate=learning_rate)
model1.compile(optimizer=adamOptimizer, metrics=['mse'])   

history_cache = model1.fit([x_train1, x_train2, y_train1],
                          verbose=1,
                          epochs=epochs,
                          batch_size=batch_size)
print('Final cost: {0:.4f}'.format(history_cache.history['loss'][-1]))



# Model 2
epochs = 500
batch_size = 10
learning_rate = 0.001
InputLayer1 = Input(shape=(1,))
InputLayer2 = Input(shape=(1,))
Layer_2 = Dense(20,activation="tanh")(Concatenate(axis=1)([InputLayer1, InputLayer2]))
OutputLayer = Dense(1, activation="linear")(Layer_2)

y_real = Input(shape=(1,))

lossF = loss_function(OutputLayer, y_real)
model2 = Model(inputs=[InputLayer1, InputLayer2, y_real], outputs=OutputLayer)
model2.add_loss(lossF)
adamOptimizer = optimizers.Adam(learning_rate=learning_rate)
model2.compile(optimizer=adamOptimizer, metrics=['mse'])   

history_cache = model2.fit([x_train1, x_train2, y_train2],
                          verbose=1,
                          epochs=epochs,
                          batch_size=batch_size)
print('Final cost: {0:.4f}'.format(history_cache.history['loss'][-1]))

