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

#产生训练集
Pilotnum=8
K = 256
N = 200000
Y = np.load('Y_'+str(Pilotnum)+'_'+str(N)+'N'+'.npy')
leibie = np.load('leibie_'+str(Pilotnum)+'_'+str(N)+'N'+'.npy')

N = Y.shape[0]
Y_mat = np.transpose(np.reshape(Y,[Y.shape[0],256,8]),[0,2,1])
Y_pilot = Y_mat[:,[0,1,4,5],:].reshape(N,1024)
Y_data  = Y_mat[:,[2,3,6,7],:].reshape(N,1024)
Y = np.concatenate((Y_pilot, Y_data), 1)

Y_mat = np.transpose(np.reshape(Y,[N,256,8]),[0,2,1])
Y_pilot = Y[:,0:1024]
Y_data  = Y[:,1024:2048].reshape(N,4,256)
Y_antena0_data = Y_data[:,0,:]+1j*Y_data[:,1,:]
Y_antena1_data = Y_data[:,2,:]+1j*Y_data[:,3,:]
y = np.zeros([N,K,2]).astype(complex)
y[:,:,0] = Y_antena0_data
y[:,:,1] = Y_antena1_data
Y_data  = Y_data.reshape(N,1024)

#训练分类
model_input = keras.Input(shape=(1024))
if Pilotnum==32:
    model_output = fenlei_32(model_input)
else:
    model_output = fenlei_8(model_input)
model = keras.Model(inputs=model_input, outputs=model_output)
model.summary()
model.compile(
    optimizer=keras.optimizers.Adam(),  # Optimizer
    # Loss function to minimize
    loss='categorical_crossentropy',
    # List of metrics to monitor
    metrics=['acc'],
)

#开始训练
model.optimizer.lr = 0.001
def scheduler(epoch, lr):
    if (epoch+1) % 150 == 0 and epoch>290:
        return lr*0.5
    else:
        return lr
print(round(model.optimizer.lr.numpy(), 5))
change_LR = tf.keras.callbacks.LearningRateScheduler(scheduler)
checkpoint = keras.callbacks.ModelCheckpoint(".model_fenlei_"+str(Pilotnum)+".best.h5", monitor='val_loss', 
                                             save_best_only=True, save_weights_only=True, mode='auto', period=1)
callbacks_list = [checkpoint,change_LR]
#
model.fit(x=Y_pilot, y=leibie, batch_size=256, epochs=2000, validation_split=0.1,callbacks=callbacks_list)

#产生训练集
Pilotnum=32
K = 256
N = 200000
Y = np.load('Y_'+str(Pilotnum)+'_'+str(N)+'N'+'.npy')
leibie = np.load('leibie_'+str(Pilotnum)+'_'+str(N)+'N'+'.npy')

N = Y.shape[0]
Y_mat = np.transpose(np.reshape(Y,[Y.shape[0],256,8]),[0,2,1])
Y_pilot = Y_mat[:,[0,1,4,5],:].reshape(N,1024)
Y_data  = Y_mat[:,[2,3,6,7],:].reshape(N,1024)
Y = np.concatenate((Y_pilot, Y_data), 1)

Y_mat = np.transpose(np.reshape(Y,[N,256,8]),[0,2,1])
Y_pilot = Y[:,0:1024]
Y_data  = Y[:,1024:2048].reshape(N,4,256)
Y_antena0_data = Y_data[:,0,:]+1j*Y_data[:,1,:]
Y_antena1_data = Y_data[:,2,:]+1j*Y_data[:,3,:]
y = np.zeros([N,K,2]).astype(complex)
y[:,:,0] = Y_antena0_data
y[:,:,1] = Y_antena1_data
Y_data  = Y_data.reshape(N,1024)

#训练分类
model_input = keras.Input(shape=(1024))
if Pilotnum==32:
    model_output = fenlei_32(model_input)
else:
    model_output = fenlei_8(model_input)
model = keras.Model(inputs=model_input, outputs=model_output)
model.summary()
model.compile(
    optimizer=keras.optimizers.Adam(),  # Optimizer
    # Loss function to minimize
    loss='categorical_crossentropy',
    # List of metrics to monitor
    metrics=['acc'],
)

#开始训练
model.optimizer.lr = 0.001
def scheduler(epoch, lr):
    if (epoch+1) % 150 == 0 and epoch>290:
        return lr*0.5
    else:
        return lr
print(round(model.optimizer.lr.numpy(), 5))
change_LR = tf.keras.callbacks.LearningRateScheduler(scheduler)
checkpoint = keras.callbacks.ModelCheckpoint(".model_fenlei_"+str(Pilotnum)+".best.h5", monitor='val_loss', 
                                             save_best_only=True, save_weights_only=True, mode='auto', period=1)
callbacks_list = [checkpoint,change_LR]
#
model.fit(x=Y_pilot, y=leibie, batch_size=256, epochs=2000, validation_split=0.1,callbacks=callbacks_list)