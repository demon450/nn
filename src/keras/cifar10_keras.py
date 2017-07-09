'''
Created on Jun 11, 2017

@author: demon
'''
import keras
import numpy as np
#from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from com.chandler.irac.datasets.Cifa10Loader import Cifa10Loader

batch_size_train = 64
num_classes = 10
epochs = 200
data_augmentation = True
reg = 0.0001

loader = Cifa10Loader();

'''
# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
'''

'''
10000x5 data 3072 pixels 10 outcomes
'''
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

'''
import sys
sys.exit("Stop here")
'''

model = Sequential()

model.add(BatchNormalization(input_shape=(32,32,3)))
#3x3 conv with 32 filters, input_shape = (32x32x3)
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(PReLU())
model.add(Conv2D(32, (3, 3)))
model.add(BatchNormalization())
model.add(PReLU())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#second conv layer
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(PReLU())
model.add(Conv2D(128, (3, 3),
                 kernel_regularizer=regularizers.l2(reg),
                 activity_regularizer=regularizers.l2(reg)))
model.add(BatchNormalization())
model.add(PReLU())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#flat to 1-D
model.add(Flatten())
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

#data pre-processing
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


#call backs
tensor_board = keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=0, batch_size=batch_size_train,
                             write_graph=True, write_grads=False, write_images=False, 
                             embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

check_point = keras.callbacks.ModelCheckpoint("saved/cifar10_epoch{epoch}.h5", monitor='val_loss', 
                                              verbose=0, save_best_only=False, save_weights_only=False, 
                                              mode='auto', period=1)

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')


#plot model graph
from keras.utils import plot_model
plot_model(model, to_file='model.png')

if not data_augmentation:
    print('Not using data augmentation.')
    history = model.fit(x_train, y_train,
              batch_size=batch_size_train,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks = [tensor_board,check_point,early_stopping],
              verbose=2)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    history = model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size_train),
                        steps_per_epoch=x_train.shape[0] // batch_size_train,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        callbacks = [tensor_board,check_point,early_stopping],
                        verbose=2)



# list all data in history
print(history.history.keys())
# summarize history for accuracy
import matplotlib.pyplot as plt

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

'''          
#model.save_weights('cifar10_weights_reg.h5')
model.save('saved/cifar10_reg.h5')
'''
