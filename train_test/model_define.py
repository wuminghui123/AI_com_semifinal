

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Reshape,GlobalAveragePooling2D,GlobalMaxPooling2D,Dense,Add,Activation,multiply


def channel_attention(input_feature, ratio=8):
	
	#channel_axis = 1 if K.image_data_format() == "channels_last" else -1
	channel = input_feature.shape[-1]
	
	shared_layer_one = Dense(channel//ratio,
							 activation='relu',
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')
	shared_layer_two = Dense(channel,
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')
	
	avg_pool = GlobalAveragePooling2D()(input_feature)    
	avg_pool = Reshape((1,1,channel))(avg_pool)
	#assert avg_pool._keras_shape[1:] == (1,1,channel)
	avg_pool = shared_layer_one(avg_pool)
	#assert avg_pool._keras_shape[1:] == (1,1,channel//ratio)
	avg_pool = shared_layer_two(avg_pool)
	#assert avg_pool._keras_shape[1:] == (1,1,channel)
	
	max_pool = GlobalMaxPooling2D()(input_feature)
	max_pool = Reshape((1,1,channel))(max_pool)
	#assert max_pool._keras_shape[1:] == (1,1,channel)
	max_pool = shared_layer_one(max_pool)
	#assert max_pool._keras_shape[1:] == (1,1,channel//ratio)
	max_pool = shared_layer_two(max_pool)
	#assert max_pool._keras_shape[1:] == (1,1,channel)
	
	cbam_feature = Add()([avg_pool,max_pool])
	cbam_feature = Activation('sigmoid')(cbam_feature)
	
#	if K.image_data_format() == "channels_first":
#		cbam_feature = Permute((3, 1, 2))(cbam_feature)
	
	return multiply([input_feature, cbam_feature])

K = 256
P_32 =32 * 2
pilotCarriers_32 = np.arange(0, K, K // P_32)
pilotCarriers_32 = tf.cast(pilotCarriers_32, dtype=tf.int32)

P_8 =8 * 2
pilotCarriers_8 = np.arange(0, K, K // P_8)
pilotCarriers_8 = tf.cast(pilotCarriers_8, dtype=tf.int32)
def MyModel_8(x):
    ini = x + 0
    #x_ini = layers.Reshape((1, 256, 8))(x)
    Y_pilot = x[:,0,:,0:4]
    Y_pilot= tf.gather(Y_pilot,pilotCarriers_8,axis=1)
    x = layers.Flatten()(Y_pilot)
    x = layers.Dense(4096, activation='linear')(x)
    x = layers.BatchNormalization()(x)
    x = Mish(x)
    x_ini = layers.Reshape((1, 256, 16))(x)
    for i in range(1):
        x = layers.Conv2D(32, (1,7), padding = 'SAME', activation='linear')(x_ini)
        x = layers.BatchNormalization()(x)
        #x = layers.LeakyReLU()(x)
        x = Mish(x)
        x = layers.Conv2D(64,(1,7), padding = 'SAME',activation="linear")(x)
        x = layers.BatchNormalization()(x)
        #x = layers.LeakyReLU()(x)
        x = Mish(x)
        x = layers.Conv2D(16, (1,7), padding = 'SAME',activation="linear")(x)
        x = layers.BatchNormalization()(x)
        x = channel_attention(x)
        x_ini = keras.layers.Add()([x_ini, x])
        #x_ini = layers.LeakyReLU()(x_ini)
        x_ini = Mish(x_ini)
    
    x_ini = tf.concat([x_ini,ini[:,:,:,4:8]],3)
    x_ini = layers.Conv2D(256, (1,7), padding = 'SAME', activation='linear')(x_ini)
    x_ini = layers.BatchNormalization()(x_ini)
    #x = layers.LeakyReLU()(x)
    x_ini = Mish(x_ini)
    for i in range(6):
        x = layers.Conv2D(512, (1,7), padding = 'SAME', activation='linear')(x_ini)
        x = layers.BatchNormalization()(x)
        #x = layers.LeakyReLU()(x)
        x = Mish(x)
        x = layers.Conv2D(1024,(1,7), padding = 'SAME',activation="linear")(x)
        x = layers.BatchNormalization()(x)
        #x = layers.LeakyReLU()(x)
        x = Mish(x)
        x = layers.Conv2D(256, (1,7), padding = 'SAME',activation="linear")(x)
        x = layers.BatchNormalization()(x)
        x = channel_attention(x)
        x_ini = keras.layers.Add()([x_ini, x])
        #x_ini = layers.LeakyReLU()(x_ini)
        x_ini = Mish(x_ini)
    x = layers.Conv2D(4, (1,7), padding = 'SAME',activation="sigmoid")(x_ini)
    x = tf.concat([x[:,:,:,0:2],x[:,:,:,2:4]],2)
    x = layers.Flatten()(x)
    return x

def MyModel_32(x):
    ini = x + 0
    #x_ini = layers.Reshape((1, 256, 8))(x)
    Y_pilot = x[:,0,:,0:4]
    Y_pilot = x[:,0,:,0:4]
    Y_pilot= tf.gather(Y_pilot,pilotCarriers_32,axis=1)
    x = layers.Flatten()(Y_pilot)
    x = layers.Dense(8192, activation='linear')(x)
    x = layers.BatchNormalization()(x)
    x = Mish(x)
    x_ini = layers.Reshape((1, 256, 32))(x)
    for i in range(1):
        x = layers.Conv2D(32, (1,5), padding = 'SAME', activation='linear')(x_ini)
        x = layers.BatchNormalization()(x)
        #x = layers.LeakyReLU()(x)
        x = Mish(x)
        x = layers.Conv2D(64,(1,5), padding = 'SAME',activation="linear")(x)
        x = layers.BatchNormalization()(x)
        #x = layers.LeakyReLU()(x)
        x = Mish(x)
        x = layers.Conv2D(32, (1,5), padding = 'SAME',activation="linear")(x)
        x = layers.BatchNormalization()(x)
        x = channel_attention(x)
        x_ini = keras.layers.Add()([x_ini, x])
        #x_ini = layers.LeakyReLU()(x_ini)
        x_ini = Mish(x_ini)
    x_ini = tf.concat([x_ini,ini[:,:,:,4:8]],3)
    x_ini = layers.Conv2D(256, (1,5), padding = 'SAME', activation='linear')(x_ini)
    x_ini = layers.BatchNormalization()(x_ini)
    #x = layers.LeakyReLU()(x)
    x_ini = Mish(x_ini)
    for i in range(6):
        x = layers.Conv2D(512, (1,5), padding = 'SAME', activation='linear')(x_ini)
        x = layers.BatchNormalization()(x)
        #x = layers.LeakyReLU()(x)
        x = Mish(x)
        x = layers.Conv2D(1024,(1,5), padding = 'SAME',activation="linear")(x)
        x = layers.BatchNormalization()(x)
        #x = layers.LeakyReLU()(x)
        x = Mish(x)
        x = layers.Conv2D(256, (1,5), padding = 'SAME',activation="linear")(x)
        x = layers.BatchNormalization()(x)
        x = channel_attention(x)
        x_ini = keras.layers.Add()([x_ini, x])
        #x_ini = layers.LeakyReLU()(x_ini)
        x_ini = Mish(x_ini)
    x = layers.Conv2D(4, (1,5), padding = 'SAME',activation="sigmoid")(x_ini)
    x = tf.concat([x[:,:,:,0:2],x[:,:,:,2:4]],2)
    x = layers.Flatten()(x)
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