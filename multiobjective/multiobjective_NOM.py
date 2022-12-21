import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Concatenate, add
from tensorflow.keras import Input
from tensorflow.keras import optimizers
from keras import backend as K
import os

_dir = "D:/Northwestern/ASU/NOM/Paper/GitHub/multiobjective/"      
save_model_name = os.path.join(_dir,'multiobjective_NN objective function1.h5')
model1 = tf.keras.models.load_model(save_model_name, compile=False)

_dir = "D:/Northwestern/ASU/NOM/Paper/GitHub/multiobjective/"      
save_model_name = os.path.join(_dir,'multiobjective_NN objective function2.h5')
model2 = tf.keras.models.load_model(save_model_name, compile=False)

xx_test1 = np.linspace(0, 10, 50)
xx_test2 = np.linspace(5, 15, 50)
xv_test, yv_test = np.meshgrid(xx_test1, xx_test2)
x_test1 = xv_test.flatten()
x_test2 = yv_test.flatten()
y_pred1 = model1.predict(list((x_test1, x_test2, x_test1)))
idx = np.argpartition(y_pred1.flatten(), 10)
print(y_pred1.flatten()[idx[:10]])

epsilon = np.linspace(0.5, 1.65, 30)
f1_ = np.linspace(0, 0, len(epsilon))
f2_ = np.linspace(0, 0, len(epsilon))
x1_ = np.linspace(0, 0, len(epsilon))
x2_ = np.linspace(0, 0, len(epsilon))
for pareto_i in range(len(epsilon)):
    print(pareto_i)
    for start_point_idx in range(10):
        
        x_optimization1 = np.array([x_test1[idx[start_point_idx]]])
        x_optimization2 = np.array([x_test2[idx[start_point_idx]]])
        
        y_optimization2 = model2.predict(list((x_optimization1, x_optimization2, x_optimization1)))
        
        def optimization_cost(output):
            return tf.reduce_mean(output)
      
        def custom_activation1(x):
            return 10 * K.relu(x-6.8) 
        
        def custom_activation2(x):
            return 10 * K.relu(x+70)
        
        def custom_activation3(x):
            return 10 * K.relu(x)
             
        def custom_activation4(x):
            return 10 * (K.relu(-x+0) + K.relu(x-10))
        
        def custom_activation5(x):
            return 10 * (K.relu(-x+5) + K.relu(x-15))
        
        def custom_activation6(x):
            return 10 * K.relu(x-epsilon[pareto_i])
        
        epochs = 2000
        batch_size = 1
        learning_rate = 0.01
        InputLayer1 = Input(shape=(1,))
        InputLayer2 = Input(shape=(1,))
        Layer_1_1 = Dense(1,activation="linear")(InputLayer1)
        Layer_1_2 = Dense(1,activation="linear")(InputLayer2)
        Layer_2_1 = Dense(20,activation="tanh", trainable=False)(Concatenate(axis=1)([Layer_1_1, Layer_1_2]))
        Layer_2_2 = Dense(20,activation="tanh", trainable=False)(Concatenate(axis=1)([Layer_1_1, Layer_1_2]))
        Layer_constraint1 = Dense(1,activation=custom_activation1, trainable=False)(Layer_1_1)
        Layer_constraint2 = Dense(1,activation=custom_activation2, trainable=False)(add(([Layer_1_1*(-8), Layer_1_2*(-4)])))
        Layer_constraint3 = Dense(1,activation=custom_activation3, trainable=False)(add(([Layer_1_1*3, Layer_1_2*(-2.5)])))
        Layer_constraint4 = Dense(1,activation=custom_activation4, trainable=False)(Layer_1_1)
        Layer_constraint5 = Dense(1,activation=custom_activation5, trainable=False)(Layer_1_2)    
        OutputLayer1 = Dense(1, activation="linear", trainable=False)(Layer_2_1)
        OutputLayer2 = Dense(1, activation=custom_activation6, trainable=False)(Layer_2_2)
        OutputLayer = add(([OutputLayer1, OutputLayer2, Layer_constraint1, Layer_constraint2, Layer_constraint3, Layer_constraint4, Layer_constraint5]))
        
        y_real = Input(shape=(1,))
        
        lossF = optimization_cost(OutputLayer)
        model_optimization = Model(inputs=[InputLayer1, InputLayer2, y_real], outputs=OutputLayer)
        model_optimization.add_loss(lossF)
        adamOptimizer = optimizers.Adam(learning_rate=learning_rate)
        model_optimization.compile(optimizer=adamOptimizer, metrics=['mse'])
        
        model_optimization.layers[10].set_weights(model1.layers[3].get_weights())
        model_optimization.layers[14].set_weights(model1.layers[5].get_weights())
        model_optimization.layers[11].set_weights(model2.layers[3].get_weights())
        model_optimization.layers[15].set_weights(model2.layers[5].get_weights())
              
        model_optimization.layers[16].set_weights([np.array([[1.]]), np.array([0.])])
        model_optimization.layers[17].set_weights([np.array([[1.]]), np.array([0.])])
        model_optimization.layers[18].set_weights([np.array([[1.]]), np.array([0.])])
        model_optimization.layers[19].set_weights([np.array([[1.]]), np.array([0.])])
        model_optimization.layers[20].set_weights([np.array([[1.]]), np.array([0.])])
             
        history_cache = model_optimization.fit([x_optimization1, x_optimization2, y_optimization2],
                                  verbose=0,
                                  epochs=epochs,
                                  batch_size=batch_size)
        
        x1 = x_optimization1 * model_optimization.layers[2].get_weights()[0] + model_optimization.layers[2].get_weights()[1]
        x2 = x_optimization2 * model_optimization.layers[3].get_weights()[0] + model_optimization.layers[3].get_weights()[1]
        f1 = model1.predict(list((x1, x2, x1)))
        f2 = model2.predict(list((x1, x2, x1)))
        # print(x1)
        # print(x2)
        # print(f1)
        # print()
        
        g1 = 70-4*x2-8*x1
        g2 = -2.5*x2+3*x1
        g3 = -6.8+x1
        g4 = -x1+0
        g5 = x1-10
        g6 = -x2+5
        g7 = x2-15
        
        if g1<=0 and g2<=0 and g3<=0 and g4<=0 and g5<=0 and g6<=0 and g7<=0:
            if f1_[pareto_i] == 0:
                f1_[pareto_i] = f1
                f2_[pareto_i] = f2
                x1_[pareto_i] = x1
                x2_[pareto_i] = x2
            else:
                if f1 < f1_[pareto_i]:
                    f1_[pareto_i] = f1
                    f2_[pareto_i] = f2
                    x1_[pareto_i] = x1
                    x2_[pareto_i] = x2
                            
    print(x1_)
    print(x2_)
    print(f1_)
    print(f2_)