""" This script demonstrates the use of a convolutional LSTM network.
This network is used to predict the next steer based on the current frame of video.
"""

import keras
import numpy as np
import skimage.io as img_io
import skimage.transform
from matplotlib import pyplot
import math
import random
from keras.models import load_model

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
    print(labels.shape)    
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
num_frames = 16
num_samples = 1
batch_size = 10
num_classes = 3


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
    
    
print(x_train.shape)
print(y_train.shape)
print(y_train[0][0])

model = load_model("saved/baby_navigator.h5")
result = model.predict(x_train, batch_size=batch_size, verbose=0)
print(result.shape)
print(y_train[0][:6])
print(result[0][:6])
