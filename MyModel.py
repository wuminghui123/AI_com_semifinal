import tensorflow as tf
from tensorflow import keras
from utils import *
import struct

from model_define import MyModel,NMSE_loss

mode=0
SNRdb=10
Pilotnum=8
###########################以下仅为信道数据载入和链路使用范例############

data1=open('H.bin','rb')
H1=struct.unpack('f'*2*2*2*32*320000,data1.read(4*2*2*2*32*320000))
H1=np.reshape(H1,[320000,2,4,32])
H=H1[:,1,:,:]+1j*H1[:,0,:,:]

Htest=H[300000:,:,:]
H=H[:300000,:,:]


#model = keras.MyModel() #定义自己的模型
# Model construction
# encoder model
model_input = keras.Input(shape=(1024))
model_output = MyModel(model_input)
model = keras.Model(inputs=model_input, outputs=model_output)

model.summary()
#model.compile(
#    optimizer=keras.optimizers.Adam(),  # Optimizer
#    # Loss function to minimize
#    loss='binary_crossentropy',
    # List of metrics to monitor
#    metrics=['binary_accuracy'],
#)
model.compile(optimizer='adam', loss=NMSE_loss)
####################使用链路和信道数据产生训练数据##########
def generator(batch,H):
    while True:
        input_labels = []
        input_samples = []
        for row in range(0, batch):
            bits0 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
            bits1 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
            X=[bits0, bits1]
            temp = np.random.randint(0, len(H))
            HH = H[temp]
            YY = MIMO(X, HH, SNRdb, mode,Pilotnum)/20 ###
            XX = np.concatenate((bits0, bits1), 0)
            input_labels.append(XX)
            input_samples.append(YY)
        batch_y = np.asarray(input_samples)
        batch_x = np.asarray(input_labels)

        Y_mat = np.transpose(np.reshape(batch_y,[batch,256,8]),[0,2,1])
        Y_pilot = Y_mat[:,[0,1,4,5],:].reshape(batch,1024)
        Y_data  = Y_mat[:,[2,3,6,7],:].reshape(batch,1024)
        batch_y = np.concatenate((Y_pilot, Y_data), 1)


        yield (batch_y, batch_x)
#####训练#########
#model.fit_generator(generator(1000,H),steps_per_epoch=50,epochs=2000)

########产生测评数据，仅供参考格式##########
def generatorXY(batch, H):
    input_labels = []
    input_samples = []
    for row in range(0, batch):
        mode = np.random.randint(0, 3)
        SNRdb = np.random.randint(0, 5)+8

        bits0 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
        bits1 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
        X = [bits0, bits1]
        temp = np.random.randint(0, len(H))
        HH = H[temp]
        YY = MIMO(X, HH, SNRdb, mode, Pilotnum) / 20  ###
        XX = np.concatenate((bits0, bits1), 0)
        input_labels.append(XX)
        input_samples.append(YY)
    batch_y = np.asarray(input_samples)
    batch_x = np.asarray(input_labels)

    Y_mat = np.transpose(np.reshape(batch_y,[batch,256,8]),[0,2,1])
    Y_pilot = Y_mat[:,[0,1,4,5],:].reshape(batch,1024)
    Y_data  = Y_mat[:,[2,3,6,7],:].reshape(batch,1024)
    batch_y = np.concatenate((Y_pilot, Y_data), 1)

    return batch_y, batch_x
#Y, X = generatorXY(100000, H)
#####训练#########
#model.fit_generator(generator(1000,H),steps_per_epoch=50,epochs=2000)
HH = np.load('H_'+str(Pilotnum)+'.npy')
Y_Pilot = np.load('Y_pilot_'+str(Pilotnum)+'.npy')
#model.fit(x=Y_Pilot, y=HH, batch_size=128, epochs=2000, validation_split=0.1)

model.optimizer.lr = 0.001
def scheduler(epoch, lr):
    if (epoch+1) % 150 == 0 and epoch>290:
        return lr*0.5
    else:
        return lr
print(round(model.optimizer.lr.numpy(), 5))
change_LR = tf.keras.callbacks.LearningRateScheduler(scheduler)

checkpoint = keras.callbacks.ModelCheckpoint(".model_"+str(Pilotnum)+".best.h5", monitor='val_loss', 
                                             save_best_only=True, save_weights_only=True, mode='auto', period=1)
callbacks_list = [checkpoint,change_LR]

model.fit(x=Y_Pilot, y=HH, batch_size=128, epochs=2000, validation_split=0.1,callbacks=callbacks_list)

#np.savetxt('Y_1.csv', Y, delimiter=',')
#X_1 = np.array(np.floor(X + 0.5), dtype=np.bool)
#X_1.tofile('X_1.bin')


