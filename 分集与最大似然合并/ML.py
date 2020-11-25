import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第二块GPU（从0开始）
import tensorflow as tf
from tensorflow import keras
import struct
import numpy as np

def rd_pre(Pilotnum,num):
    if Pilotnum==8:
        index = 2
    else:
        index = 1
    data1=open('X_pre_'+str(index)+'_'+str(num)+'.bin','rb')
    H1=struct.unpack('b'*1024*10000,data1.read(1024*10000))
    H1=np.reshape(H1,[10000,1024])
    return H1
X_2 = np.zeros([10000,1024])
for i in range(5):
    X_2 = X_2+rd_pre(8,i)
X_2[X_2<3]=0
X_2[X_2>2]=1
X_2 = np.array(np.floor(X_2 + 0.5), dtype=np.bool)    
X_2.tofile('X_pre_2.bin')
print(np.sum(X_2 != rd_pre(8,0))/(1024*10000))

X_1 = np.zeros([10000,1024])
for i in range(5):
    X_1 = X_1+rd_pre(32,i)
X_1[X_1<3]=0
X_1[X_1>2]=1
X_1 = np.array(np.floor(X_1 + 0.5), dtype=np.bool)    
X_1.tofile('X_pre_1.bin')
print(np.sum(X_1 != rd_pre(32,1))/(1024*10000))