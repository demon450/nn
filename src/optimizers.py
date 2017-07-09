'''
Created on Jun 10, 2017

@author: demon
'''
import numpy as np

class Optimizers():
    def __call__(self, shape, dtype=None):
        raise NotImplementedError

    """another way of declaring constructor"""
    @classmethod
    def from_config(config):
        return config
    
class SgdCrossEntropy(Optimizers):
   
    def __call__(self, X, y, output):
        # compute the gradient on scores
        dscores = output
        dscores[range(output.shape[0]), y] -= 1
        dscores /= output.shape[0]
          
        # backpropate the gradient to the parameters (W,b)
        dW = np.dot(X.T, dscores)
        db = np.sum(dscores, axis=0, keepdims=True)      
        
        return dW, db, dscores

class SgdReLU(Optimizers):
   
    def __call__(self, X, y, output):
        output[y <= 0] = 0
        # finally into W,b
        dW = np.dot(X.T, output)
        db = np.sum(output, axis=0, keepdims=True)
        
        return dW, db, output

class Sgd(Optimizers):
    def __call__(self, probs, f, x):
        """ 
        a naive implementation of numerical gradient of f at x 
        - f should be a function that takes a single argument
        - x is the point (numpy array) to evaluate the gradient at
        """ 
        fx = f(x)  # evaluate function value at original point
        grad = np.zeros(x.shape)
        h = 0.00001
        
        # iterate over all indexes in x
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:        
            # evaluate function at x+h
            ix = it.multi_index
            old_value = x[ix]
            x[ix] = old_value + h  # increment by h
            fxh = f(x)  # evalute f(x + h)
            x[ix] = old_value  # restore to previous value (very important!)
            
            # compute the partial derivative
            grad[ix] = (fxh - fx) / h  # the slope
            it.iternext()  # step to next dimension
    
        return grad
