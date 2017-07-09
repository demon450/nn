import numpy as np
import matplotlib.pyplot as plt

def load_data(N=100, D=2, K=3):
    '''    
    :param N:number of points PER class
    :param D:dimension/column
    :param K:number of classes
    '''
    X = np.zeros((N * K, D))  # data matrix (each row = single example)
    y = np.zeros(N * K, dtype='uint8')  # class labels
    for j in range(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
   
    visualize(X, y)
    
    return X, y

# X,y=load_data()
# print(y)

def visualize(X, y):
    # lets visualize the data:
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40)  # , cmap=plt.cm.Spectral)
    plt.show()
    

