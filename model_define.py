

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#def MyModel(x,bits):

    
    #x = layers.Dense(2048, activation='linear')(x)
    #x = layers.BatchNormalization()(x)
    #x = layers.LeakyReLU()(x)
    #x = layers.Dense(4096, activation='linear')(x)
    #x = layers.BatchNormalization()(x)
    #x = layers.LeakyReLU()(x)
    #x = layers.Dense(bits, activation='sigmoid')(x)

    #return x

def MyModel(x):
    ini = x + 0
    #x_ini = layers.Reshape((1, 256, 8))(x)
    Y_pilot = x[:,0,:,0:4]
    x = layers.Flatten()(Y_pilot)
    x = layers.Dense(4096, activation='linear')(x)
    x = layers.BatchNormalization()(x)
    x = Mish(x)
    x_ini = layers.Reshape((1, 256, 16))(x)
    for i in range(2):
        x = layers.Conv2D(512, (1,7), padding = 'SAME', activation='linear')(x_ini)
        x = layers.BatchNormalization()(x)
        x = Mish(x)
        x = layers.Conv2D(1024,(1,7), padding = 'SAME',activation="linear")(x)
        x = layers.BatchNormalization()(x)
        x = Mish(x)
        x = layers.Conv2D(16, (1,7), padding = 'SAME',activation="linear")(x)
        x = layers.BatchNormalization()(x)
        x_ini = keras.layers.Add()([x_ini, x])
        x_ini = Mish(x_ini)
    x_ini = tf.concat([x_ini,ini[:,:,:,4:8]],3)
    for i in range(2):
        x = layers.Conv2D(512, (1,7), padding = 'SAME', activation='linear')(x_ini)
        x = layers.BatchNormalization()(x)
        #x = layers.LeakyReLU()(x)
        x = Mish(x)
        x = layers.Conv2D(1024,(1,7), padding = 'SAME',activation="linear")(x)
        x = layers.BatchNormalization()(x)
        #x = layers.LeakyReLU()(x)
        x = Mish(x)
        x = layers.Conv2D(20, (1,7), padding = 'SAME',activation="linear")(x)
        x = layers.BatchNormalization()(x)
        x_ini = keras.layers.Add()([x_ini, x])
        #x_ini = layers.LeakyReLU()(x_ini)
        x_ini = Mish(x_ini)
    x = layers.Flatten()(x_ini)
    x = layers.Dense(units=1024, activation='sigmoid')(x)
    return x



def NMSE_loss(x, x_hat):
    x_real = tf.reshape(x[:, :, :, 0], (tf.shape(x)[0], -1))
    x_imag = tf.reshape(x[:, :, :, 1], (tf.shape(x)[0], -1))
    x_hat_real = tf.reshape(x_hat[:, :, :, 0], (tf.shape(x_hat)[0], -1))
    x_hat_imag = tf.reshape(x_hat[:, :, :, 1], (tf.shape(x_hat)[0], -1))
    power = tf.reduce_sum(x_real**2 + x_imag**2, axis=1)
    mse = tf.reduce_sum((x_real - x_hat_real) ** 2 + (x_imag - x_hat_imag) ** 2, axis=1)
    nmse = tf.reduce_mean(mse / power)
    return nmse

def Mish(x):
    return x*tf.nn.tanh(tf.nn.softplus(x))