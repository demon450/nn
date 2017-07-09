from keras.models import load_model
import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import pylab as plt


model_file = "saved/cnn_lstm.h5"
movie_file = "dataset/cnn_lstm.npy"
shifted_file = "dataset/cnn_lstm_shifted.npy"

model = load_model(model_file)
noisy_movies = np.load(movie_file)
shifted_movies = np.load(shifted_file)

which = 1004
track = noisy_movies[which][:7, ::, ::, ::]

# And then compare the predictions
# to the ground truth
track2 = noisy_movies[which][::, ::, ::, ::]
for i in range(10):
    fig = plt.figure(figsize=(10, 5))

    ax = fig.add_subplot(121)

    if i >= 7:
        ax.text(1, 3, 'Predictions !', fontsize=20, color='w')
    else:
        ax.text(1, 3, 'Inital trajectory', fontsize=20)

    toplot = track[i, ::, ::, 0]

    plt.imshow(toplot)
    ax = fig.add_subplot(122)
    plt.text(1, 3, 'Ground truth', fontsize=20)

    toplot = track2[i, ::, ::, 0]
    if i >= 2:
        toplot = shifted_movies[which][i - 1, ::, ::, 0]

    plt.imshow(toplot)
    plt.savefig('%i_animate.png' % (i + 1))
    
    

which = 1004
track = noisy_movies[which][:7, ::, ::, ::]



from keras.models import Model
layer_name = 'my_layer'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(track[np.newaxis, ::, ::, ::, ::])