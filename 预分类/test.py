import tensorflow as tf
import numpy as np
from tensorflow import keras
from utils import *
import struct
from tensorflow.keras import layers
from math import *

def Mish(x):
    return x*tf.nn.tanh(tf.nn.softplus(x))
def fenlei_8(x):
    x = layers.Dense(512, activation='linear')(x)
    x = layers.BatchNormalization()(x)
    x = Mish(x)
    x = layers.Dense(256, activation='linear')(x)
    x = layers.BatchNormalization()(x)
    x = Mish(x)
    x = layers.Dense(3, activation='softmax')(x)
    return x
def fenlei_32(x):
    x = layers.Dense(512, activation='linear')(x)
    x = layers.BatchNormalization()(x)
    x = Mish(x)
    x = layers.Dense(256, activation='linear')(x)
    x = layers.BatchNormalization()(x)
    x = Mish(x)
    x = layers.Dense(3, activation='softmax')(x)
    return x

Y_1 = np.loadtxt('Y_1.csv', delimiter=',')
Y_2 = np.loadtxt('Y_2.csv', delimiter=',')

Pilotnum = 8
model_input = keras.Input(shape=(1024))
if Pilotnum==32:
    model_output = fenlei_32(model_input)
else:
    model_output = fenlei_8(model_input)
model = keras.Model(inputs=model_input, outputs=model_output)
#model.summary()
model.compile(
    optimizer=keras.optimizers.Adam(),  # Optimizer
    # Loss function to minimize
    loss='categorical_crossentropy',
    # List of metrics to monitor
    metrics=['acc'],
)
model.load_weights(".model_fenlei_"+str(Pilotnum)+".best.h5")
if Pilotnum==32:
    Y = Y_1
else:
    Y = Y_2
N = Y.shape[0]
Y_mat = np.transpose(np.reshape(Y,[Y.shape[0],256,8]),[0,2,1])
Y_pilot = Y_mat[:,[0,1,4,5],:].reshape(N,1024)
Y_data  = Y_mat[:,[2,3,6,7],:].reshape(N,1024)
Y = np.concatenate((Y_pilot, Y_data), 1)
Y_mat = np.transpose(np.reshape(Y,[N,256,8]),[0,2,1])
Y_pilot = Y[:,0:1024]
X_out = model.predict(Y_pilot, batch_size=128)
d=[c.argmax() for c in X_out]
n = 0
for i in range(10000):
    if d[i]==0:
        n=n+1
print('8导频的mode0数量为',n)

Pilotnum = 32
model_input = keras.Input(shape=(1024))
if Pilotnum==32:
    model_output = fenlei_32(model_input)
else:
    model_output = fenlei_8(model_input)
model = keras.Model(inputs=model_input, outputs=model_output)
#model.summary()
model.compile(
    optimizer=keras.optimizers.Adam(),  # Optimizer
    # Loss function to minimize
    loss='categorical_crossentropy',
    # List of metrics to monitor
    metrics=['acc'],
)
model.load_weights(".model_fenlei_"+str(Pilotnum)+".best.h5")
if Pilotnum==32:
    Y = Y_1
else:
    Y = Y_2
N = Y.shape[0]
Y_mat = np.transpose(np.reshape(Y,[Y.shape[0],256,8]),[0,2,1])
Y_pilot = Y_mat[:,[0,1,4,5],:].reshape(N,1024)
Y_data  = Y_mat[:,[2,3,6,7],:].reshape(N,1024)
Y = np.concatenate((Y_pilot, Y_data), 1)
Y_mat = np.transpose(np.reshape(Y,[N,256,8]),[0,2,1])
Y_pilot = Y[:,0:1024]
X_out = model.predict(Y_pilot, batch_size=128)
d=[c.argmax() for c in X_out]
n = 0
for i in range(10000):
    if d[i]==0:
        n=n+1
print('32导频的mode0数量为',n)

