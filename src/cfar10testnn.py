import numpy as np
from com.chandler.irac.datasets.Cifa10Loader import Cifa10Loader
from com.chandler.irac.test.NearestNeighbor import NearestNeighbor

loader = Cifa10Loader();
nn = NearestNeighbor();

Xtr, Ytr, Xte, Yte = loader.load_cifar10('../../../data/CIFAR10/')

# print(Xtr.shape)
# flatten out all images to be one-dimensional
# Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072
# Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) # Xte_rows becomes 10000 x 3072

# print(Xtr_rows)

size = 1150;
if(size == 0):
    size = Xte.shape[0];

# slice test data    
Xtr = Xtr[:size]
Ytr = Ytr[:size]
Xte = Xte[:size]
Yte = Yte[:size]

nn = NearestNeighbor()  # create a Nearest Neighbor classifier class
nn.train(Xtr, Ytr)  # train the classifier on the training images and labels
Yte_predict = nn.predict(Xte)  # predict labels on the test images
# and now print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)
right_predict = [Yte_predict[i] for i in range(len(Yte_predict)) if Yte_predict[i] == Yte[i]]
print ('accuracy: %f' % (np.mean(Yte_predict == Yte)))  # calculate the (right predict/lenth of Yte)
print ('accuracy: %f' % (len(right_predict) / size))
