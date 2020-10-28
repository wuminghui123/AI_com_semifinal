

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
    x = layers.Dense(256, activation='linear')(x)
    #x = layers.BatchNormalization()(x)
    #x = layers.LeakyReLU()(x)
    x_ini = layers.Reshape((4, 32, 2))(x)
    #x_ini = tf.reshape(x, (tf.shape(x)[0],4,4,64))

    for i in range(2):
        x = layers.Conv2D(128, 3, padding = 'SAME', activation='linear')(x_ini)
        x = layers.BatchNormalization()(x)
        #x = layers.LeakyReLU()(x)
        x = Mish(x)
        x = layers.Conv2D(256,3, padding = 'SAME',activation="linear")(x)
        x = layers.BatchNormalization()(x)
        #x = layers.LeakyReLU()(x)
        x = Mish(x)
        x = layers.Conv2D(2, 3, padding = 'SAME',activation="linear")(x)
        x = layers.BatchNormalization()(x)
        x_ini = keras.layers.Add()([x_ini, x])
        #x_ini = layers.LeakyReLU()(x_ini)
        x_ini = Mish(x_ini)


    decoder_output = layers.Conv2D(2, 3, padding = 'SAME',activation="linear")(x_ini)

    return decoder_output



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