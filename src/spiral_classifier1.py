'''
Created on Jun 9, 2017

@author: demon
'''
from com.chandler.irac.initializers import Random
from com.chandler.irac.losses import CrossEntropy
from com.chandler.irac.optimizers import SgdCrossEntropy
from com.chandler.irac.regularizations import L2
from com.chandler.irac.datasets.spiralpoints import *



'''
100x3 data points with 2 features(x,y coordinate) and 3 possible outcomes
'''
X, y = load_data(100, 2, 3);

# some hyperparameters
step_size = 1e-0  # learning rate
reg = 1e-3  # regularization strength
scale = 0.01  # W,b scale

# init W b
init = Random();
W, b = init(2, 3, scale);

# init sgdce,reg,loss classes
sgdce = SgdCrossEntropy();
l2 = L2();
ce = CrossEntropy();

# gradient descent loop
for i in range(1000):
    # compute class scores
    scores = np.dot(X, W) + b  
    # calculate loss
    loss, probs = ce(scores, X.shape[0], y, reg)
    # claculate regulated loss
    reg_loss = l2(W, reg) + loss
    if i % 10 == 0:
        print ("iteration %d: loss %f" % (i, loss))
    # back propagation
    dW, db, dscores = sgdce(X, y, probs)
    dW += reg * W  # regularization gradient
          
    # perform a parameter update
    W += -step_size * dW
    b += -step_size * db

# evaluate training set accuracy
scores = np.dot(X, W) + b
predicted_class = np.argmax(scores, axis=1)
print ('training accuracy: %.2f' % (np.mean(predicted_class == y)))


