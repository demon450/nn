'''
Created on Jun 9, 2017

@author: demon
'''
from com.chandler.irac.test.initializers import Random
from com.chandler.irac.test.losses import CrossEntropy
from com.chandler.irac.test.optimizers import SgdCrossEntropy, SgdReLU
from com.chandler.irac.test.regularizations import L2
from com.chandler.irac.datasets.spiralpoints import *
from com.chandler.irac.test.activations import ReLu



'''
100x3 data points with 2 features(x,y coordinate) and 3 possible outcomes
:param N:number of points PER class
:param D:dimension/column
:param K:number of classes
'''
N = 100
D = 2
K = 3

X, y = load_data(N, D, K);

# some hyperparameters
step_size = 1e-0  # learning rate
reg = 1e-3  # regularization strength
h = 100  # size of hidden layer
scale = 0.01  # W,b scale

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
for i in range(5000):
    # hidden layer activation
    hidden_layer = relu(W, b, X);
    # compute class scores for output layer from hidden layer
    scores = np.dot(hidden_layer, W2) + b2 
    # calculate loss
    loss, probs = ce(scores, X.shape[0], y, reg)
    # calculate regulated loss for hidden and output layer
    reg_loss = l2(W, reg) + l2(W2, reg) + loss
    if i % 10 == 0:
        print ("iteration %d: loss %f" % (i, loss))
        
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
X, y = load_data(N, D, K);
hidden_layer = np.maximum(0, np.dot(X, W) + b)
scores = np.dot(hidden_layer, W2) + b2
predicted_class = np.argmax(scores, axis=1)
print ('training accuracy: %.2f' % (np.mean(predicted_class == y)))


