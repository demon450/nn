'''
Created on Jun 9, 2017

@author: demon

No Mnist data loader!!!!! not functional
'''
from com.chandler.irac.test.initializers import Random
from com.chandler.irac.test.losses import CrossEntropy
from com.chandler.irac.test.optimizers import SgdCrossEntropy, SgdReLU
from com.chandler.irac.test.regularizations import L2

from com.chandler.irac.test.activations import ReLu
import numpy as np



'''
:param N:number of images
:param D:number of pixels
:param K:number of classes
'''

D = 784
K = 10

'''
'''
#print('x_train shape:', x_train.shape)
#print('y_train shape:', y_train.shape)
Xtr = np.reshape(x_train, (x_train.shape[0],28*28))
Ytr = y_train
Xte = np.reshape(x_test, (x_test.shape[0],28*28))
Yte = y_test
#print('x_train shape:', Xtr.shape)
#print('y_train shape:', Ytr.shape)

'''
import sys
sys.exit("Stop here")
'''

# some hyperparameters
step_size = 0.001  # learning rate
reg = 0.0001  # regularization strength
h = 60  # size of hidden layer
scale = 0.001  # W,b scale
batch_size_train = 64  # how many images to process for each loop
epochs = 200

# init W b for hidden layer
init = Random();
W, b = init(D, h, scale);
# init W2 b2 for output layer
W2, b2 = init(h, K, scale);

# init sgdce,reg,loss classes
sgdce = SgdCrossEntropy()
sgdrelu = SgdReLU()
l2 = L2();
ce = CrossEntropy();
relu = ReLu()


# gradient descent loop
loop_size = int(Xtr.shape[0] / batch_size_train)
for j in range(epochs):
    if(j<9):
        step_size = step_size - step_size/(j+2)
    elif(j<50):
        step_size = step_size - step_size/(j*30)
    else:
        step_size = step_size - step_size/(j*70)
        
                
    print(step_size)
    for i in range(loop_size):
        X = Xtr[i * batch_size_train:(i + 1) * batch_size_train]
        y = Ytr[i * batch_size_train:(i + 1) * batch_size_train]
    
        # hidden layer activation
        hidden_layer = relu(W, b, X);
        # compute class scores for output layer from hidden layer
        scores = np.dot(hidden_layer, W2) + b2 
        # print(scores)
        # calculate loss
        loss, probs = ce(scores, X.shape[0], y, reg)
        # calculate regulated loss for hidden and output layer
        reg_loss = l2(W, reg) + l2(W2, reg) + loss
        if i % 100 == 0:
            print ("epoch %d iteration %d: loss %f" % (j, i, reg_loss))
            
        # back propagation to hidden layer
        dW2, db2, dscores = sgdce(hidden_layer, y, probs)
        # next backprop into hidden layer
        dhidden = np.dot(dscores, W2.T)
        dW, db, output = sgdrelu(X, hidden_layer, dhidden)
        # add regularization gradient contribution
        dW2 += reg * W2
        dW += reg * W
          
        # perform a parameter update
        W += -step_size * dW
        b += -step_size * db
        W2 += -step_size * dW2
        b2 += -step_size * db2
    
    
    # evaluate training set accuracy
    hidden_layer = np.maximum(0, np.dot(Xte, W) + b)
    scores = np.dot(hidden_layer, W2) + b2
    predicted_class = np.argmax(scores, axis=1)
    print ('training accuracy: %.2f' % (np.mean(predicted_class == Yte)))


