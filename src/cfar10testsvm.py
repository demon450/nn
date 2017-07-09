import numpy as np


loader = Cifa10Loader();
CrossEntropy = Losses();

Xtr, Ytr, Xte, Yte = loader.load_cifar10('../../../data/CIFAR10/')

size = 16;
if(size == 0):
    size = Xte.shape[0];

# slice test data    
Xtr = Xtr[:size]
Ytr = Ytr[:size]
Xte = Xte[:size]
Yte = Yte[:size]

# to use the generic code above we want a function that takes a single argument
# (the weights in our case) so we close over X_train and Y_train
def CIFAR10_loss_fun(W):
    return CrossEntropy.SvmLoss(Xtr, Ytr, W)

gd = GD();
W = np.random.rand(10, 3072) * 0.001  # random weight vector
df = gd.eval_numerical_gradient(CIFAR10_loss_fun, W)  # get the gradient
loss_original = CIFAR10_loss_fun(W)  # the original CrossEntropy
print ('original CrossEntropy: %f' % loss_original)

# lets see the effect of multiple step sizes
for step_size_log in [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1]:
    step_size = 10 ** step_size_log
    W_new = W - step_size * df  # new position in the weight space
    loss_new = CIFAR10_loss_fun(W_new)
    print ('for step size %f new CrossEntropy: %f' % (step_size, loss_new))
    
    
