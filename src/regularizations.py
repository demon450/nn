'''
Created on Jun 10, 2017

@author: demon
'''
import numpy as np

class Regularizations():
    def __call__(self, shape, dtype=None):
        raise NotImplementedError

    """another way of declaring constructor"""
    @classmethod
    def from_config(config):
        return config
    
class L2(Regularizations):
   
    def __call__(self, W, reg):
        return 0.5*reg*np.sum(W*W)