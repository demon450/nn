import numpy as np

class Activations():
    def __call__(self, shape, dtype=None):
        raise NotImplementedError

    """another way of declaring constructor"""
    @classmethod
    def from_config(config):
        return config
    
class ReLu(Activations):
   
    def __call__(self, W, b, X):
        output = np.maximum(0, np.dot(X, W) + b) 
        return output
    
   
    
class SoftMax(Activations):    
    def __call__(self,z=np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0])):
        z_exp = np.exp(z);
        sum_z_exp = np.sum(z_exp, 0,np.float64)
        
        softmax = z_exp/sum_z_exp
        print(softmax)
        return softmax
    
#a = Activation();
#a.soft_max_v()    