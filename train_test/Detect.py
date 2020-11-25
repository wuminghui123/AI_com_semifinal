import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第二块GPU（从0开始）
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from utils import *
import struct
import numpy as np
from model_define import MyModel_8,MyModel_32,NMSE_loss

def decode(Y,Pilotnum):
    model_input = keras.Input(shape=(1,256,8))
    if Pilotnum==8:
        model_output = MyModel_8(model_input)
    else:
        model_output = MyModel_32(model_input)
    model = keras.Model(inputs=model_input, outputs=model_output)

    model.compile(
        optimizer=keras.optimizers.Adam(),  # Optimizer
        # Loss function to minimize
        loss='binary_crossentropy',
        # List of metrics to monitor
        metrics=['binary_accuracy'],
    )
    N = Y.shape[0]
    Y_mat = np.transpose(np.reshape(Y,[N,256,8]),[0,2,1])
    Y = np.reshape(Y,[N,1,256,8])

    Y_t = np.zeros([N,1,256,8])
    Y_t[:,0,:,0] = Y_mat[:,0,:]
    Y_t[:,0,:,1] = Y_mat[:,1,:]
    Y_t[:,0,:,2] = Y_mat[:,4,:]
    Y_t[:,0,:,3] = Y_mat[:,5,:]

    Y_t[:,0,:,4] = Y_mat[:,2,:]
    Y_t[:,0,:,5] = Y_mat[:,3,:]
    Y_t[:,0,:,6] = Y_mat[:,6,:]
    Y_t[:,0,:,7] = Y_mat[:,7,:]
    Y = Y_t
    model.load_weights(".model_"+str(Pilotnum)+".best.h5")

    X_out = model.predict(Y, batch_size=128)
    
    X_pre = np.array(np.floor(X_out + 0.5), dtype=np.bool)
    return X_pre

    ###########################以下仅为信道数据载入和链路使用范例############

data1=open('H_val.bin','rb')
H1=struct.unpack('f'*2*2*2*32*2000,data1.read(4*2*2*2*32*2000))
H1=np.reshape(H1,[2000,2,4,32])
H=H1[:,1,:,:]+1j*H1[:,0,:,:]


def generatorXY(batch, H,SNR,Pilotnum):
    input_labels = []
    input_samples = []
    input_H = []
    for row in range(0, batch):
        #mode = np.random.randint(0, 3)
        mode = 0
        SNRdb = 10
        bits0 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
        bits1 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
        X = [bits0, bits1]
        temp = np.random.randint(0, len(H))
        HH = H[temp]
        YY = MIMO(X, HH, SNRdb, mode, Pilotnum) / 20  ###
        XX = np.concatenate((bits0, bits1), 0)
        input_labels.append(XX)
        input_samples.append(YY)
        input_H.append(HH)
    batch_y = np.asarray(input_samples)
    batch_x = np.asarray(input_labels)
    batch_H = np.asarray(input_H)
    return batch_y, batch_x
N = 10000

Y, X = generatorXY(N, H,8,8)
X_pre = decode(Y,8)
acc = np.sum(X_pre == X)/N/1024
print('Pilot8 : The accuracy is',acc)

Y, X = generatorXY(N, H,8,32)
X_pre = decode(Y,32)
acc = np.sum(X_pre == X)/N/1024
print('Pilot32 : The accuracy is',acc)

Y_1 = np.loadtxt('Y_1.csv', delimiter=',')
Y_2 = np.loadtxt('Y_2.csv', delimiter=',')

X_pre_1 = decode(Y_1,32)
X_pre_2 = decode(Y_2,8)

X_pre_1.tofile('X_pre_1.bin')
X_pre_2.tofile('X_pre_2.bin')