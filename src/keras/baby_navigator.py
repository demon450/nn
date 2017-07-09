""" This script demonstrates the use of a convolutional LSTM network.
This network is used to predict the next steer based on the current frame of video.
"""

'''
import os
from sys import implementation
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
'''

import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.layers.pooling import MaxPool3D
import numpy as np
import skimage.io as img_io
import skimage.transform
from matplotlib import pyplot
import math
import random

'''
Image processing
'''
def show_image(img):
    pyplot.figure()
    #pyplot.subplot(rows,columns,1)
    pyplot.imshow(img)
    pyplot.axis('on')
    pyplot.title('Original image = RGB')
    pyplot.show()

def show_images(imgs,labels):    
    dim = round(math.sqrt(imgs.shape[0]))
    pyplot.figure()
    for i in range(imgs.shape[0]):
        pyplot.subplot(dim,dim,i+1)
        pyplot.imshow(imgs[i])
        pyplot.axis('on')
        if(labels[i]==1):
            pyplot.title("Left:"+str(labels[i]))
        elif(labels[i] == 0):
            pyplot.title("Straight:"+str(labels[i]))
        else:
            pyplot.title("Right:"+str(labels[i]))
    pyplot.show()
    

def load_image(url,preview=True):
    img = skimage.img_as_float(img_io.imread(url)).astype(np.float32)
    print("Image shape:")
    print(img.shape)
    
    if(preview):
        return show_image(img)
    else:
        return img
    
def resize_constain(img, short_edge):
    ratio = img.shape[0]/img.shape[1]
    
    if(ratio>1):
        img_small = skimage.transform.resize(img, (round(short_edge*ratio), short_edge))
    else:
        img_small = skimage.transform.resize(img, (short_edge,round(short_edge/ratio)))
    
    return img_small


def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]


#img = resize_constain(img, int(img_size))
#show_image(crop_center(img_resized,128,128))


'''
Functions to move forward or backward
'''
def move(img, step):
    return crop_center(img,img.shape[1]-step,img.shape[0]-step)


'''
Functions to generate different directions for one frame
'''
def noise(img):
    #noise_f = np.random.randn(*img.shape)/300
    #img+=noise_f
    return img

#show_image(noise(img_resized))
   
def steer(img, amount):
    img = np.swapaxes(img, 0, 1)
    from collections import deque
    items = deque(img)
    
    items.rotate(int(-amount))    
    items = np.swapaxes(items, 0, 1)
    return items

#show_image(crop_center(steer(img_resized,10),128,128))
           
def rotate(img, angle):
    img_rotated = skimage.transform.rotate(img, angle)    
    return img_rotated

#show_image(crop_center(rotate(img_resized,-5),128,128))


def random_direction(img, direction=None):
    #steer left or right
    factor = (random.randint(0, 8)/100)*((-1)**random.randrange(2))
    '''
    if(direction is None):
        factor = (random.randint(3, 8)/100)*((-1)**random.randrange(2))
    elif(direction == 1):
        factor = (random.randint(3, 8)/100)
    else:
        factor = -(random.randint(3, 8)/100)
    '''
    
    steer_angle = int(img.shape[1]*factor)
    rotation_angle = random.randint(0, 9)*((-1)**random.randrange(2))
    img_p = steer(img,steer_angle)
    img_p = rotate(img_p,rotation_angle)
    img_p = noise(img_p)
    img_p = resize_constain(img_p, int(img_size))
    img_p = crop_center(img_p,img_size,img_size)
    
    #print(factor)
    #facing right, turn left=1
    if(factor > 0):
        label = 1
    elif(factor == 0):
        label = 0
    else:
        label = 2
    return img_p,label
    
    
#load image
img_size = 64
num_frames = 9
num_samples = 1000
batch_size = 10
num_classes = 3
train_file = 'dataset/baby_navigator_video.npy'
label_file = 'dataset/baby_navigator_label.npy'

import os
if not (os.path.exists(train_file)):
    img = load_image("../images/yard1.jpg", False)
    img = resize_constain(img, 128)
    #1000 samples 16 frames
    x_train = np.zeros((num_samples,num_frames,img_size,img_size,3))
    y_train = np.zeros((num_samples,num_frames,num_classes), dtype=int)
    for j in range(num_samples):
        imgs = np.zeros((num_frames,img_size,img_size,3))
        labels = np.zeros(num_frames, dtype=int)
        #move forward, create 10 frames
        for i in range(num_frames):
            move_step = int(img.shape[0]*0.03)
            img_p = move(img, move_step*(i+1))
            img_temp,labels[i] = random_direction(img_p)
            imgs[i,:,:,:] = img_temp[np.newaxis,:,:,:] 
            
            x_train[j,:,:,:,:] = imgs[np.newaxis,:,:,:,:]
            y_train[j,i,labels[i]] = 1
            #y_train[(j*num_frames)+i,labels[i]] = 1
        if(j==2):
            show_images(imgs,labels)
        print(j)        
    
    np.save(train_file, x_train)
    np.save(label_file, y_train)
    print("video prepared!")

x_train = np.load(train_file)
y_train = np.load(label_file)
print(x_train.shape)
print(y_train.shape)
print(y_train[0][0])

model = Sequential()
model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
                   input_shape=(None, img_size, img_size, 3),
                   padding='valid', return_sequences=True,
                   implementation=2))
model.add(BatchNormalization())
model.add(MaxPool3D((1,2,2), strides=(1,2,2)))
model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
                   padding='valid', return_sequences=True,
                   implementation=2))
model.add(BatchNormalization())
model.add(MaxPool3D((1,2,2), strides=(1,2,2)))
'''
model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
model.add(BatchNormalization())

model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
model.add(BatchNormalization())

model.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
               activation='sigmoid',
               padding='same', data_format='channels_last'))
'''

model.add(TimeDistributed(Flatten()))
model.add(TimeDistributed(Dense(512)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(TimeDistributed(Dense(num_classes)))#, input_shape=(10, 16)))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.01, decay=1e-6)

model.compile(loss='binary_crossentropy', optimizer=opt)
#plot model graph
from keras.utils import plot_model
plot_model(model, to_file='baby_navigator.png', show_shapes=True)


#call backs
tensor_board = keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=0, batch_size=batch_size,
                             write_graph=True, write_grads=False, write_images=False, 
                             embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

check_point = keras.callbacks.ModelCheckpoint("saved/baby_navigator.h5", monitor='val_loss', 
                                              verbose=0, save_best_only=False, save_weights_only=False, 
                                              mode='auto', period=1)

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
# Train the network
#y_train = y_train.ravel()
history = model.fit(x_train, y_train, batch_size=batch_size,
        epochs=100, validation_split=0.2, verbose = 1,
        shuffle=True,
        callbacks = [tensor_board,check_point,early_stopping],)


# list all data in history
print(history.history.keys())


# summarize history for loss
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'test'], loc='upper left')
pyplot.show()

