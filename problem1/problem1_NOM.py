import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Concatenate, add, multiply
from tensorflow.keras import Input
from tensorflow.keras import optimizers
from keras import backend as K
import os

_dir = ".../problem1/"      
save_model_name = os.path.join(_dir,'problem1_NN objective function.h5')
model = tf.keras.models.load_model(save_model_name, compile=False)

print("Weights and biases of the layers before training the model: \n")
for layer in model.layers:
    print(layer.name)
    print(layer.get_weights())

xx_test1 = np.linspace(13, 20, 50)
xx_test2 = np.linspace(0, 20, 50)
xv_test, yv_test = np.meshgrid(xx_test1, xx_test2)
X_test1 = xv_test.flatten()
X_test2 = yv_test.flatten()
x_test1 = (X_test1[(X_test1>=13) & (X_test1<=20) & (X_test2>=0) & (X_test2<=20) & (-(X_test1-5)**2-(X_test2-5)**2+100<=0) & ((X_test1-6)**2-(X_test2-5)**2-100<=0)])/20
x_test2 = (X_test2[(X_test1>=13) & (X_test1<=20) & (X_test2>=0) & (X_test2<=20) & (-(X_test1-5)**2-(X_test2-5)**2+100<=0) & ((X_test1-6)**2-(X_test2-5)**2-100<=0)])/20
y_pred = model.predict(list((x_test1, x_test2, x_test1))) *1000
idx = np.argpartition(y_pred.flatten(), 10)
print(y_pred.flatten()[idx[:10]])


for start_point_idx in range(5):
    x_optimization1 = np.array([x_test1[idx[start_point_idx]]])
    x_optimization2 = np.array([x_test2[idx[start_point_idx]]])
    y_optimization = model.predict(list((x_optimization1, x_optimization2, x_optimization1)))
    
    def optimization_cost(output):
        return tf.reduce_mean(output)
   
    def custom_activation1(x):
        return 10 * (K.relu(-20*x+13) + K.relu(20*x-20))
    
    def custom_activation2(x):
        return 10 * (K.relu(-20*x+0) + K.relu(20*x-20))
    
    def custom_activation3(x):
        return 10 * K.relu(-x+100)   
    
    def custom_activation4(x):
        return 10 * K.relu(x-100)
    
    epochs = 2000
    batch_size = 1
    learning_rate = 0.01
    InputLayer1 = Input(shape=(1,))
    InputLayer2 = Input(shape=(1,))
    Layer_NOM_1_1 = Dense(1,activation="linear", trainable=False)(InputLayer1)
    Layer_NOM_1_2 = Dense(1,activation="linear", trainable=False)(InputLayer2)
    Layer_1_1 = Dense(1,activation="linear")(Layer_NOM_1_1)
    Layer_1_2 = Dense(1,activation="linear")(Layer_NOM_1_2)
    Layer_2 = Dense(20,activation="tanh", trainable=False)(Concatenate(axis=1)([Layer_1_1, Layer_1_2]))
    Layer_constraint1 = Dense(1,activation=custom_activation1, trainable=False)(Layer_1_1)
    Layer_constraint2 = Dense(1,activation=custom_activation2, trainable=False)(Layer_1_2)
    Layer_constraint3 = Dense(1,activation=custom_activation3, trainable=False)(add(([multiply(([20*Layer_1_1-5, 20*Layer_1_1-5])), multiply(([20*Layer_1_2-5, 20*Layer_1_2-5]))])))
    Layer_constraint4 = Dense(1,activation=custom_activation4, trainable=False)(add(([multiply(([20*Layer_1_1-6, 20*Layer_1_1-6])), multiply(([20*Layer_1_2-5, 20*Layer_1_2-5]))])))
    OutputLayer1 = Dense(1, activation="linear", trainable=False)(Layer_2)
    OutputLayer = add(([OutputLayer1, Layer_constraint1, Layer_constraint2, Layer_constraint3, Layer_constraint4]))
    
    y_real = Input(shape=(1,))
    
    lossF = optimization_cost(OutputLayer)
    model_optimization = Model(inputs=[InputLayer1, InputLayer2, y_real], outputs=OutputLayer)
    model_optimization.add_loss(lossF)
    adamOptimizer = optimizers.Adam(learning_rate=learning_rate)
    model_optimization.compile(optimizer=adamOptimizer, metrics=['mse'])
    
    
    model_optimization.layers[27].set_weights(model.layers[3].get_weights())
    model_optimization.layers[30].set_weights(model.layers[5].get_weights())
    
    model_optimization.layers[2].set_weights([np.array([[1.]]), np.array([0.])])
    model_optimization.layers[3].set_weights([np.array([[1.]]), np.array([0.])])
  
    model_optimization.layers[4].set_weights([np.array([[1.]]), np.array([0.])])
    model_optimization.layers[5].set_weights([np.array([[1.]]), np.array([0.])])
        
    model_optimization.layers[31].set_weights([np.array([[1.]]), np.array([0.])])
    model_optimization.layers[32].set_weights([np.array([[1.]]), np.array([0.])])
    model_optimization.layers[33].set_weights([np.array([[1.]]), np.array([0.])])
    model_optimization.layers[34].set_weights([np.array([[1.]]), np.array([0.])])
         
    history_cache = model_optimization.fit([x_optimization1, x_optimization2, y_optimization],
                              verbose=0,
                              epochs=epochs,
                              batch_size=batch_size)
    print('Final cost: {0:.4f}'.format(history_cache.history['loss'][-1]))    
    
    InputLayer = Input(shape=(1,))
    Layer_NOM_1 = Dense(1,activation="linear")(InputLayer)
    OutputLayer = Dense(1,activation="linear")(Layer_NOM_1)
    model_xopt = Model(inputs=InputLayer, outputs=OutputLayer)
    model_xopt.compile(optimizer='adam', loss='categorical_crossentropy')
    
    model_xopt.layers[1].set_weights(model_optimization.layers[2].get_weights())
    model_xopt.layers[2].set_weights(model_optimization.layers[4].get_weights()) 
    x1_opt_ = model_xopt.predict(list((x_optimization1)))
    
    model_xopt.layers[1].set_weights(model_optimization.layers[3].get_weights())
    model_xopt.layers[2].set_weights(model_optimization.layers[5].get_weights()) 
    x2_opt_ = model_xopt.predict(list((x_optimization2)))
    
    x1_opt = 20* x1_opt_
    x2_opt = 20* x2_opt_
    print(x1_opt)
    print(x2_opt)
    
    print(1000*model.predict(list((x1_opt_[0], x2_opt_[0], x1_opt_[0]))))
    print(((x1_opt-10)**3 + (x2_opt-20)**3))
    print(x1_opt>=13 and x1_opt<=20)
    print(x2_opt>=0 and x2_opt<=20)
    print(-(x1_opt-5)**2-(x2_opt-5)**2+100<=0)
    print((x1_opt-6)**2+(x2_opt-5)**2-100<=0)
    print()
    