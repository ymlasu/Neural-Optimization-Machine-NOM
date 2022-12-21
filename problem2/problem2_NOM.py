import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Concatenate, add, subtract, multiply
from tensorflow.keras import Input
from tensorflow.keras import optimizers
from keras import backend as K
import os


_dir = ".../problem2/"      
save_model_name = os.path.join(_dir,'problem2_NN objective function.h5')
model = tf.keras.models.load_model(save_model_name, compile=False)

print("Weights and biases of the layers before training the model: \n")
for layer in model.layers:
    print(layer.name)
    print(layer.get_weights())

xx_test1 = np.linspace(0, 1.5, 50)
xx_test2 = np.linspace(-1, 1, 50)
xv_test, yv_test = np.meshgrid(xx_test1, xx_test2)
X_test1 = xv_test.flatten()
X_test2 = yv_test.flatten()
x_test1 = X_test1[(X_test1>=0) & (X_test1<=1.5) & (X_test2>=-1) & (X_test2<=1) & (X_test1-X_test2<=0.5) & (X_test1*X_test2<=15)]
x_test2 = X_test2[(X_test1>=0) & (X_test1<=1.5) & (X_test2>=-1) & (X_test2<=1) & (X_test1-X_test2<=0.5) & (X_test1*X_test2<=15)]
y_pred = model.predict(list((x_test1, x_test2, x_test1)))
idx = np.argpartition(y_pred.flatten(), 10)
print(y_pred.flatten()[idx[:10]])

for start_point_idx in range(10):
    
    x_optimization1 = np.array([x_test1[idx[start_point_idx]]])
    x_optimization2 = np.array([x_test2[idx[start_point_idx]]])
    
    y_optimization = model.predict(list((x_optimization1, x_optimization2, x_optimization1)))
    
    def optimization_cost(output):
        return tf.reduce_mean(output)
   
    def custom_activation1(x):
        return 10 * (K.relu(-x-0) + K.relu(x-1.5))
    
    def custom_activation2(x):
        return 10 * (K.relu(-(x-(-1))) + K.relu(x-1))

    def custom_activation3(x):
        return 10 * K.relu(-(x-0.5))
    
    def custom_activation4(x):
        return 10 * K.relu(x-15)    
    
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
    Layer_constraint3 = Dense(1,activation=custom_activation3, trainable=False)(subtract(([Layer_1_1*1, Layer_1_2])))
    Layer_constraint4 = Dense(1,activation=custom_activation4, trainable=False)(multiply(([Layer_1_1, Layer_1_2])))
    OutputLayer1 = Dense(1, activation="linear", trainable=False)(Layer_2)
    OutputLayer = add(([OutputLayer1, Layer_constraint1, Layer_constraint2, Layer_constraint3, Layer_constraint4]))
    
    y_real = Input(shape=(1,))
    
    lossF = optimization_cost(OutputLayer)
    model_optimization = Model(inputs=[InputLayer1, InputLayer2, y_real], outputs=OutputLayer)
    model_optimization.add_loss(lossF)
    adamOptimizer = optimizers.Adam(learning_rate=learning_rate)
    model_optimization.compile(optimizer=adamOptimizer, metrics=['mse'])
    

    model_optimization.layers[8].set_weights(model.layers[3].get_weights())
    model_optimization.layers[11].set_weights(model.layers[5].get_weights())
    
    model_optimization.layers[2].set_weights([np.array([[1.]]), np.array([0.])]) #no hidden neuron for NOM
    model_optimization.layers[3].set_weights([np.array([[1.]]), np.array([0.])]) #no hidden neuron for NOM  
  
    model_optimization.layers[4].set_weights([np.array([[1.]]), np.array([0.])]) #initial weights 1
    model_optimization.layers[5].set_weights([np.array([[1.]]), np.array([0.])]) #initial weights 1
    
    model_optimization.layers[12].set_weights([np.array([[1.]]), np.array([0.])])
    model_optimization.layers[13].set_weights([np.array([[1.]]), np.array([0.])])
    model_optimization.layers[14].set_weights([np.array([[1.]]), np.array([0.])])
    model_optimization.layers[15].set_weights([np.array([[1.]]), np.array([0.])])
         
    history_cache = model_optimization.fit([x_optimization1, x_optimization2, y_optimization],
                              verbose=0,
                              epochs=epochs,
                              batch_size=batch_size)    
    
    
    InputLayer = Input(shape=(1,))
    Layer_NOM_1 = Dense(1,activation="linear")(InputLayer)
    OutputLayer = Dense(1,activation="linear")(Layer_NOM_1)
    model_xopt = Model(inputs=InputLayer, outputs=OutputLayer)
    model_xopt.compile(optimizer='adam', loss='categorical_crossentropy')
    
    model_xopt.layers[1].set_weights(model_optimization.layers[2].get_weights())
    model_xopt.layers[2].set_weights(model_optimization.layers[4].get_weights()) 
    x1_opt = model_xopt.predict(list((x_optimization1)))
    
    model_xopt.layers[1].set_weights(model_optimization.layers[3].get_weights())
    model_xopt.layers[2].set_weights(model_optimization.layers[5].get_weights()) 
    x2_opt = model_xopt.predict(list((x_optimization2)))
    
    
    print(x1_opt)
    print(x2_opt)
    
    print(model.predict(list((x1_opt[0], x2_opt[0], x1_opt[0]))))
    print(-10 * (np.cos(x1_opt*x2_opt) - x1_opt*x2_opt/100 - np.sin((x1_opt+x2_opt)*(x1_opt+x2_opt))))
    print(x1_opt>=0 and x1_opt<=1.5)
    print(x2_opt>=-1 and x2_opt<=1)
    print(x1_opt-x2_opt>=0.5)
    print(x1_opt*x2_opt<=15)
    print()
    
    tf.keras.backend.clear_session()
