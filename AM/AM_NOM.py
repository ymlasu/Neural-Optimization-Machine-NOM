import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Concatenate, add, multiply
from tensorflow.keras import Input
from tensorflow.keras import optimizers
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation


AM = pd.read_csv(".../AM/AM data.csv")
S = AM['Sa']
R = AM['R']
speed = AM['Speed']
power = AM['Power']
hatch = AM['Hatch']
thickness = AM['Thickness']
temperature = AM['Heat temperature']
time = AM['Heat time']
train_data_S = S
train_data_R = R
train_data_speed = speed
train_data_power = power
train_data_hatch = hatch
train_data_thickness = thickness
train_data_temperature = temperature
train_data_time = time
train_data_S_min = train_data_S.min(axis=0)
train_data_S_max = train_data_S.max(axis=0)
train_data_S_range = train_data_S_max - train_data_S_min
train_data_R_min = train_data_R.min(axis=0)
train_data_R_max = train_data_R.max(axis=0)
train_data_R_range = train_data_R_max - train_data_R_min


def mdn_cost(mu, sigma, y, indx):
    dist = tf.distributions.Normal(loc=mu, scale=sigma)
    return tf.reduce_mean(-dist.log_prob(y)*indx-dist.log_survival_function(y)*(tf.constant([[1.0]])-indx))
 

from keras.constraints import Constraint

class NonPos(Constraint):
    def __call__(self, w):
        return w * K.cast(K.less_equal(w, 0.), K.floatx())

class GreaterThanMu0(Constraint):
    def __call__(self, w):
        constraint_row_0 = w[0, :] * K.cast(K.greater_equal(w[0, :], 0.), K.floatx())
        constraint_row_0 = K.expand_dims(constraint_row_0, axis=0)
        constraint_row_1 = w[1, :] * K.cast(K.greater_equal(w[1, :], 0.), K.floatx())
        constraint_row_1 = K.expand_dims(constraint_row_1, axis=0)
        constraint_row_2 = w[2, :] * K.cast(K.greater_equal(w[2, :], 0.), K.floatx())
        constraint_row_2 = K.expand_dims(constraint_row_2, axis=0)
        constraint_row_3 = w[3, :] * K.cast(K.greater_equal(w[3, :], 0.), K.floatx())
        constraint_row_3 = K.expand_dims(constraint_row_3, axis=0)
        constraint_row_4 = w[4, :] * K.cast(K.greater_equal(w[4, :], 0.), K.floatx())
        constraint_row_4 = K.expand_dims(constraint_row_4, axis=0)
        constraint_row_5 = w[5, :] * K.cast(K.greater_equal(w[5, :], 0.), K.floatx())
        constraint_row_5 = K.expand_dims(constraint_row_5, axis=0)
        constraint_row_6 = w[6, :] * K.cast(K.greater_equal(w[6, :], 0.), K.floatx())
        constraint_row_6 = K.expand_dims(constraint_row_6, axis=0)
        constraint_row_7 = w[7, :] * K.cast(K.greater_equal(w[7, :], 0.), K.floatx())
        constraint_row_7 = K.expand_dims(constraint_row_7, axis=0)
        constraint_row_8 = w[8, :] * K.cast(K.greater_equal(w[8, :], 0.), K.floatx())
        constraint_row_8 = K.expand_dims(constraint_row_8, axis=0)
        constraint_row_9 = w[9, :] * K.cast(K.greater_equal(w[9, :], 0.), K.floatx())
        constraint_row_9 = K.expand_dims(constraint_row_9, axis=0)
        full_w = K.concatenate([constraint_row_0, constraint_row_1, constraint_row_2, constraint_row_3, constraint_row_4,
                                constraint_row_5, constraint_row_6, constraint_row_7, constraint_row_8, constraint_row_9,
                                w[10:15, :]], axis=0)
        return full_w 
    
class GreaterThanSigma(Constraint):
    def __call__(self, w):
        constraint_row_0 = w[0, :] * K.cast(K.greater_equal(w[0, :], 0.), K.floatx())
        constraint_row_0 = K.expand_dims(constraint_row_0, axis=0)
        constraint_row_1 = w[1, :] * K.cast(K.greater_equal(w[1, :], 0.), K.floatx())
        constraint_row_1 = K.expand_dims(constraint_row_1, axis=0)
        constraint_row_2 = w[2, :] * K.cast(K.greater_equal(w[2, :], 0.), K.floatx())
        constraint_row_2 = K.expand_dims(constraint_row_2, axis=0)
        constraint_row_3 = w[3, :] * K.cast(K.greater_equal(w[3, :], 0.), K.floatx())
        constraint_row_3 = K.expand_dims(constraint_row_3, axis=0)
        constraint_row_4 = w[4, :] * K.cast(K.greater_equal(w[4, :], 0.), K.floatx())
        constraint_row_4 = K.expand_dims(constraint_row_4, axis=0)
        full_w = K.concatenate([constraint_row_0, constraint_row_1, constraint_row_2, constraint_row_3, constraint_row_4,
                                w[5:15, :]], axis=0)
        return full_w 
    


def activation_sigma(x):
    return K.elu(x) + 1

get_custom_objects().update({'activation_sigma': Activation(activation_sigma)})


_dir = ".../AM/AM_NN objective function"      
model = tf.keras.models.load_model(_dir, custom_objects={'NonPos': NonPos, 'GreaterThanMu0':GreaterThanMu0, 'activation_sigma':activation_sigma, 'GreaterThanSigma':GreaterThanSigma}, compile=False)



grid = 6
xx_speed = np.linspace(0, 1, grid)
xx_power = np.linspace(0, 1, grid)
xx_hatch = np.linspace(0, 1, grid)
xx_thickness = np.linspace(0, 1, grid)
xx_temperature = np.linspace(0, 1, grid)
xx_time = np.linspace(0, 1, grid)

test_data_speed, test_data_power, test_data_hatch, test_data_thickness, test_data_temperature, test_data_time \
= np.meshgrid(xx_speed, xx_power, xx_hatch, xx_thickness, xx_temperature, xx_time)

test_data_speed = test_data_speed.flatten()
test_data_power = test_data_power.flatten()
test_data_hatch = test_data_hatch.flatten()
test_data_thickness = test_data_thickness.flatten()
test_data_temperature = test_data_temperature.flatten()
test_data_time = test_data_time.flatten()

test_data_S = np.repeat((520 - train_data_S_min) / train_data_S_range, len(test_data_speed))
test_data_R = np.repeat((0.1 - train_data_R_min) / train_data_R_range, len(test_data_speed))
test_missing_indx = pd.Series(np.linspace(1, 1, len(test_data_speed)))

test_data = list((test_data_S, test_data_R, 
              test_data_speed, test_data_power, test_data_hatch, test_data_thickness, test_data_temperature, test_data_time,
              test_missing_indx, test_missing_indx, test_missing_indx, test_missing_indx, test_missing_indx, test_missing_indx))

mu_pred, sigma_pred = model.predict(test_data)   
    
 
len(test_data_speed)

idx = np.argpartition(-mu_pred.flatten(), 10)
print(mu_pred.flatten()[idx[:10]])



def optimization_cost(output):
    return tf.reduce_mean(output)
  
def custom_activation1(x):
    return 10 * (K.relu(-x) + K.relu(x-1))


epochs = 2000
batch_size = 1
learning_rate = 0.001
start_speed = Input(shape=(1,))
start_power = Input(shape=(1,))
start_hatch = Input(shape=(1,))
start_thickness = Input(shape=(1,))
start_temperature = Input(shape=(1,))
start_time = Input(shape=(1,))

InputLayer_S = Input(shape=(1,))
InputLayer_R = Input(shape=(1,))
InputLayer_speed = Dense(1,activation="linear")(start_speed)
InputLayer_power = Dense(1,activation="linear")(start_power)
InputLayer_hatch = Dense(1,activation="linear")(start_hatch)
InputLayer_thickness = Dense(1,activation="linear")(start_thickness)
InputLayer_temperature = Dense(1,activation="linear")(start_temperature)
InputLayer_time = Dense(1,activation="linear")(start_time)

InputMask_speed = Input(shape=(1,))
InputMask_power = Input(shape=(1,))
InputMask_hatch = Input(shape=(1,))
InputMask_thickness = Input(shape=(1,))
InputMask_temperature = Input(shape=(1,))
InputMask_time = Input(shape=(1,))

SelectiveInputLayer_speed = Dense(1,activation="tanh", trainable=False)(InputLayer_speed)
SelectiveInputLayer_power = Dense(1,activation="tanh", trainable=False)(InputLayer_power)
SelectiveInputLayer_hatch = Dense(1,activation="tanh", trainable=False)(InputLayer_hatch)
SelectiveInputLayer_thickness = Dense(1,activation="tanh", trainable=False)(InputLayer_thickness)
SelectiveInputLayer_temperature = Dense(1,activation="tanh", trainable=False)(InputLayer_temperature)
SelectiveInputLayer_time = Dense(1,activation="tanh", trainable=False)(InputLayer_time)

multiplied_speed = multiply([SelectiveInputLayer_speed, InputMask_speed])
multiplied_power = multiply([SelectiveInputLayer_power, InputMask_power])
multiplied_hatch = multiply([SelectiveInputLayer_hatch, InputMask_hatch])
multiplied_thickness = multiply([SelectiveInputLayer_thickness, InputMask_thickness])
multiplied_temperature = multiply([SelectiveInputLayer_temperature, InputMask_temperature])
multiplied_time = multiply([SelectiveInputLayer_time, InputMask_time])

Layer_1_1 = Dense(5,activation="tanh", trainable=False)(InputLayer_S)
Layer_1_2 = Dense(5,activation="tanh", trainable=False)(InputLayer_S)
Layer_1_3 = Dense(5,activation="tanh", trainable=False)(InputLayer_R)
Layer_1_4 = Dense(5,activation="tanh", trainable=False)(InputLayer_R) 
Layer_1_5 = Dense(5,activation="tanh", trainable=False)(Concatenate(axis=1)([multiplied_speed, multiplied_power, multiplied_hatch, multiplied_thickness, multiplied_temperature, multiplied_time]))
Layer_1_6 = Dense(5,activation="tanh", trainable=False)(Concatenate(axis=1)([multiplied_speed, multiplied_power, multiplied_hatch, multiplied_thickness, multiplied_temperature, multiplied_time]))
merged_Layer_1_135 = Concatenate()([Layer_1_1, Layer_1_3, Layer_1_5])
merged_Layer_1_246 = Concatenate()([Layer_1_2, Layer_1_4, Layer_1_6])
mu0 = Dense(1, activation="linear", trainable=False)(merged_Layer_1_135)
sigma = Dense(1, activation=activation_sigma, trainable=False)(merged_Layer_1_246)
merged_mu0_sigma = Concatenate(axis=1)([mu0,sigma])
mu = Dense(1, activation="linear", trainable=False)(merged_mu0_sigma)

Layer_constraint_speed = Dense(1,activation=custom_activation1, trainable=False)(InputLayer_speed)
Layer_constraint_power = Dense(1,activation=custom_activation1, trainable=False)(InputLayer_power)
Layer_constraint_hatch = Dense(1,activation=custom_activation1, trainable=False)(InputLayer_hatch)
Layer_constraint_thickness = Dense(1,activation=custom_activation1, trainable=False)(InputLayer_thickness)
Layer_constraint_temperature = Dense(1,activation=custom_activation1, trainable=False)(InputLayer_temperature)
Layer_constraint_time = Dense(1,activation=custom_activation1, trainable=False)(InputLayer_time)

OutputLayer = add(([-1*mu, 1.96*sigma, Layer_constraint_speed, Layer_constraint_power, Layer_constraint_hatch, Layer_constraint_thickness, Layer_constraint_temperature, Layer_constraint_time]))

lossF = optimization_cost(OutputLayer)
model_optimization = Model(inputs=[InputLayer_S, InputLayer_R, 
                                   start_speed, start_power, start_hatch, start_thickness, start_temperature, start_time, 
                                   InputMask_speed, InputMask_power, InputMask_hatch, InputMask_thickness, InputMask_temperature, InputMask_time], 
                           outputs=OutputLayer)
model_optimization.add_loss(lossF)
adamOptimizer = optimizers.Adam(learning_rate=learning_rate)
model_optimization.compile(optimizer=adamOptimizer,metrics=['mse'])

model_optimization.layers[12].set_weights(model.layers[6].get_weights())
model_optimization.layers[14].set_weights(model.layers[8].get_weights())
model_optimization.layers[16].set_weights(model.layers[10].get_weights())
model_optimization.layers[18].set_weights(model.layers[12].get_weights())
model_optimization.layers[20].set_weights(model.layers[14].get_weights())
model_optimization.layers[22].set_weights(model.layers[16].get_weights())
model_optimization.layers[34].set_weights(model.layers[28].get_weights())
model_optimization.layers[35].set_weights(model.layers[29].get_weights())
model_optimization.layers[36].set_weights(model.layers[30].get_weights())
model_optimization.layers[37].set_weights(model.layers[31].get_weights())
model_optimization.layers[38].set_weights(model.layers[32].get_weights())
model_optimization.layers[39].set_weights(model.layers[33].get_weights())
model_optimization.layers[42].set_weights(model.layers[36].get_weights())
model_optimization.layers[43].set_weights(model.layers[37].get_weights())
model_optimization.layers[45].set_weights(model.layers[39].get_weights())

model_optimization.layers[48].set_weights([np.array([[1.]]), np.array([0.])])
model_optimization.layers[49].set_weights([np.array([[1.]]), np.array([0.])])
model_optimization.layers[50].set_weights([np.array([[1.]]), np.array([0.])])
model_optimization.layers[51].set_weights([np.array([[1.]]), np.array([0.])])
model_optimization.layers[52].set_weights([np.array([[1.]]), np.array([0.])])
model_optimization.layers[53].set_weights([np.array([[1.]]), np.array([0.])])


for start_point_idx in range(10):
    start_S = np.array([test_data_S[idx[start_point_idx]]])
    start_R = np.array([test_data_R[idx[start_point_idx]]])
    start_speed = np.array([test_data_speed[idx[start_point_idx]]])
    start_power = np.array([test_data_power[idx[start_point_idx]]])
    start_hatch = np.array([test_data_hatch[idx[start_point_idx]]])
    start_thickness = np.array([test_data_thickness[idx[start_point_idx]]])
    start_temperature = np.array([test_data_temperature[idx[start_point_idx]]])
    start_time = np.array([test_data_time[idx[start_point_idx]]])
    start_missing_indx = np.array([test_missing_indx[idx[start_point_idx]]])
    optimization_start = [start_S, start_R, 
                  start_speed, start_power, start_hatch, start_thickness, start_temperature, start_time,
                  start_missing_indx, start_missing_indx, start_missing_indx, start_missing_indx, start_missing_indx, start_missing_indx]   
       
    history_cache = model_optimization.fit(optimization_start,
                              verbose=0,
                              epochs=epochs,
                              batch_size=batch_size)
    print('Final cost: {0:.4f}'.format(history_cache.history['loss'][-1]))
        
    
    speed_optimal_norm = start_speed * model_optimization.layers[6].get_weights()[0] + model_optimization.layers[6].get_weights()[1]
    power_optimal_norm = start_power * model_optimization.layers[7].get_weights()[0] + model_optimization.layers[7].get_weights()[1]
    hatch_optimal_norm = start_hatch * model_optimization.layers[8].get_weights()[0] + model_optimization.layers[8].get_weights()[1]
    thickness_optimal_norm = start_thickness * model_optimization.layers[9].get_weights()[0] + model_optimization.layers[9].get_weights()[1]
    temperature_optimal_norm = start_temperature * model_optimization.layers[10].get_weights()[0] + model_optimization.layers[10].get_weights()[1]
    time_optimal_norm = start_time * model_optimization.layers[11].get_weights()[0] + model_optimization.layers[11].get_weights()[1]

    
    train_data_speed_min = train_data_speed.min(axis=0)
    train_data_speed_max = train_data_speed.max(axis=0)
    train_data_speed_range = train_data_speed_max - train_data_speed_min
    print(speed_optimal_norm  * train_data_speed_range + train_data_speed_min)
    
    train_data_power_min = train_data_power.min(axis=0)
    train_data_power_max = train_data_power.max(axis=0)
    train_data_power_range = train_data_power_max - train_data_power_min
    print(power_optimal_norm  * train_data_power_range + train_data_power_min)
    
    train_data_hatch_min = train_data_hatch.min(axis=0)
    train_data_hatch_max = train_data_hatch.max(axis=0)
    train_data_hatch_range = train_data_hatch_max - train_data_hatch_min
    print(hatch_optimal_norm  * train_data_hatch_range + train_data_hatch_min)
    
    train_data_thickness_min = train_data_thickness.min(axis=0)
    train_data_thickness_max = train_data_thickness.max(axis=0)
    train_data_thickness_range = train_data_thickness_max - train_data_thickness_min
    print(thickness_optimal_norm  * train_data_thickness_range + train_data_thickness_min)
    
    train_data_temperature_min = train_data_temperature.min(axis=0)
    train_data_temperature_max = train_data_temperature.max(axis=0)
    train_data_temperature_range = train_data_temperature_max - train_data_temperature_min
    print(temperature_optimal_norm  * train_data_temperature_range + train_data_temperature_min)
    
    train_data_time_min = train_data_time.min(axis=0)
    train_data_time_max = train_data_time.max(axis=0)
    train_data_time_range = train_data_time_max - train_data_time_min
    print(time_optimal_norm  * train_data_time_range + train_data_time_min)
    
    optimization_data = [test_data_S[np.newaxis].T[0], test_data_R[np.newaxis].T[0], 
          speed_optimal_norm, power_optimal_norm, hatch_optimal_norm, thickness_optimal_norm, temperature_optimal_norm, time_optimal_norm,
          start_missing_indx, start_missing_indx, start_missing_indx, start_missing_indx, start_missing_indx, start_missing_indx]
    
    mu_opt, sigma_opt = model.predict(optimization_data)
    print(mu_opt)
    print(sigma_opt)
    print(mu_opt-1.96*sigma_opt)
    print()
    