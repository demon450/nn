import numpy as np

class NearestNeighbor():
    def __init__(self):
        pass

    def train(self, X, y):
        # the nearest neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        num_test = X.shape[0]
        # lets make sure that the output type matches the input type
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)
        # loop over all test rows
        for i in range(num_test):
        # find the nearest training image to the i'th test image
        # using the L1 distance (sum of absolute value differences)
        # using broadcasting to find the ith(min index) in Xtr(trainning set)
            distances = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)
            min_index = np.argmin(distances)  # get the index with smallest distance
            # print('ith'+str(i)+' min index:'+str(min_index))
            Ypred[i] = self.ytr[min_index]  # copy label from label set, Ypred is supposed to have correct label now
    
        return Ypred


