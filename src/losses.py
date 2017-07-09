import numpy as np

class Losses():
    def __call__(self, shape, dtype=None):
        raise NotImplementedError

    """another way of declaring constructor"""
    @classmethod
    def from_config(config):
        return config
    
class SvmLoss(Losses):
    """
    A faster half-vectorized implementation. half-vectorized
    refers to the fact that for a single example the implementation contains
    no for loops, but there is still one loop over the examples (outside this function)
    """        
    def __call__(self, x, y, W):
        delta = 1.0
        scores = W.dot(np.transpose(x))
        scores = np.transpose(scores)
        # print(scores.shape)
        # print(y.shape)
        # compute the margins for all classes in one vector operation
        margins = np.maximum(0, scores - scores[y] + delta)
        # on y-th position scores[y] - scores[y] canceled and gave delta. We want
        # to ignore the y-th position and only consider margin on max wrong class
        margins[y] = 0
        loss_i = np.sum(margins)
        return loss_i
    
class SvmLossVector(Losses):
        """
        fully-vectorized implementation :
        - X holds all the training examples as columns (e.g. 3073 x 50,000 in CIFAR-10)
        - y is array of integers specifying correct class (e.g. 50,000-D array)
        - W are weights (e.g. 10 x 3073)
        """
        # evaluate CrossEntropy over all examples in X without using any for loops
        # left as exercise to reader in the assignment
        
class CrossEntropy(Losses):
    
    def __call__(self, scores, num_examples, y, reg):
        '''
        :param y: right data INDEX 1-D array
        '''
        # get unnormalized probabilities
        exp_scores = np.exp(scores)
        # normalize them for each example
        output = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        corect_logprobs = -np.log(output[range(num_examples), y])
        # compute the CrossEntropy: average cross-entropy CrossEntropy and regularization
        loss = np.sum(corect_logprobs) / num_examples
        
        return loss, output
        
