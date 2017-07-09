import keras
import numpy as np
from keras.models import load_model
from com.chandler.irac.datasets.Cifa10Loader import Cifa10Loader

batch_size = 64
num_classes = 10



'''
10000x5 data 3072 pixels 10 outcomes
'''
loader = Cifa10Loader()
x_train, y_train, x_test, y_test = loader.load_cifar10('../../../../../data/CIFAR10/', 5)
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)

#reshape data
x_train = x_train.reshape(x_train.shape[0], 3, 32, 32)
x_test = x_test.reshape(x_test.shape[0], 3, 32, 32)

#swap axis
x_train = np.swapaxes(x_train, 1, 2)
x_train = np.swapaxes(x_train, 2, 3)
x_test = np.swapaxes(x_test, 1, 2)
x_test = np.swapaxes(x_test, 2, 3)
        
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)


# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
#data pre-processing
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255




#load model
model = load_model('saved/cifar10_reg.h5')
#plot model graph
from keras.utils import plot_model
plot_model(model, to_file='load_model.png')


scores = model.evaluate(x_test, y_test, batch_size=batch_size)
print("Evaluation Result:")
print(model.metrics_names)
print(scores)



