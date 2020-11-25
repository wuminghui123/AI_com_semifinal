import tensorflow as tf
import numpy as np
from tensorflow import keras
from utils import *
import struct
from tensorflow.keras import layers
from math import *



#生成训练集
def generatorXY(batch, H):
    input_labels = []
    input_samples = []
    input_leibie = []
    input_H = []
    for row in range(0, batch):
        if row%10000==0:
            print(row)
        mode = np.random.randint(0, 3)
        SNRdb = np.random.randint(0, 5)+8
        leibie_n = np.zeros(3)
        leibie_n[mode]=1
        bits0 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
        bits1 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
        X = [bits0, bits1]
        temp = np.random.randint(0, len(H))
        HH = H[temp]
        YY = MIMO(X, HH, SNRdb, mode, Pilotnum) / 20  ###
        XX = np.concatenate((bits0, bits1), 0)
        input_labels.append(XX)
        input_samples.append(YY)
        input_leibie.append(leibie_n)
        input_H.append(HH)
    batch_y = np.asarray(input_samples)
    batch_x = np.asarray(input_labels)
    batch_leibie = np.asarray(input_leibie)
    batch_H = np.asarray(input_H)
    out_H = np.zeros([batch,4,32,2])
    out_H[:,:,:,0] = np.real(batch_H)
    out_H[:,:,:,1] = np.imag(batch_H)
    return batch_y, batch_x, batch_leibie,out_H

data1=open('H.bin','rb')
H1=struct.unpack('f'*2*2*2*32*320000,data1.read(4*2*2*2*32*320000))
H1=np.reshape(H1,[320000,2,4,32])
H=H1[:,1,:,:]+1j*H1[:,0,:,:]
Htest=H[300000:,:,:]
H=H[:300000,:,:]
K = 256
N = 200000
Pilotnum=32
Y, X, leibie, H_est = generatorXY(N, H)
np.save('X_'+str(Pilotnum)+'_'+str(N)+'N'+'.npy',X)
np.save('Y_'+str(Pilotnum)+'_'+str(N)+'N'+'.npy',Y)
np.save('leibie_'+str(Pilotnum)+'_'+str(N)+'N'+'.npy',leibie)
np.save('Hest_'+str(Pilotnum)+'_'+str(N)+'N'+'.npy',H_est)

Y = 0
X = 0
leibie = 0
Pilotnum=8
Y, X, leibie, H_est = generatorXY(N, H)
np.save('X_'+str(Pilotnum)+'_'+str(N)+'N'+'.npy',X)
np.save('Y_'+str(Pilotnum)+'_'+str(N)+'N'+'.npy',Y)
np.save('leibie_'+str(Pilotnum)+'_'+str(N)+'N'+'.npy',leibie)
np.save('Hest_'+str(Pilotnum)+'_'+str(N)+'N'+'.npy',H_est)