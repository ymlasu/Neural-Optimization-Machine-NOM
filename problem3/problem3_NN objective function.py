import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras import Input
from tensorflow.keras import optimizers


xx_train1 = np.linspace(.1, 1, 100)
xx_train2 = np.linspace(.1, 7, 100)
xv, yv = np.meshgrid(xx_train1, xx_train2)
x_train1 = xv.flatten()
x_train2 = yv.flatten()
y_train = (np.sin(2*x_train1))**3 * np.sin(2*x_train2) / (x_train1**3*(x_train1+x_train2))


def loss_function(output, y):
    return tf.reduce_mean(tf.math.square(output - y))

epochs = 500
batch_size = 10
learning_rate = 0.01
InputLayer1 = Input(shape=(1,))
InputLayer2 = Input(shape=(1,))
Layer_2 = Dense(20,activation="tanh")(Concatenate(axis=1)([InputLayer1, InputLayer2]))
OutputLayer = Dense(1, activation="linear")(Layer_2)

y_real = Input(shape=(1,))

lossF = loss_function(OutputLayer, y_real)
model = Model(inputs=[InputLayer1, InputLayer2, y_real], outputs=OutputLayer)
model.add_loss(lossF)
adamOptimizer = optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=adamOptimizer, metrics=['mse'])   

history_cache = model.fit([x_train1, x_train2, y_train],
                          verbose=1,
                          epochs=epochs,
                          batch_size=batch_size)
print('Final cost: {0:.4f}'.format(history_cache.history['loss'][-1]))
