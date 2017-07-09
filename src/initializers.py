'''
Created on Jun 9, 2017

@author: demon
'''

import numpy as np

class Initializer(object):
    """Initializer base class: all initializers inherit from this class.
    """
    def __call__(self, shape, dtype=None):
        raise NotImplementedError

    """another way of declaring constructor"""
    @classmethod
    def from_config(config):
        return config


class Random(Initializer):
    """Initializer that generates random W and b.
    """    
    def __call__(self, D, K, scale):
        '''
        :param D: dimension - number of features for one data sample
        :param K: number of class/outcome
        '''
        # initialize parameters randomly
        W = scale * np.random.randn(D, K)
        b = np.zeros((1, K))
        
        return W, b;
