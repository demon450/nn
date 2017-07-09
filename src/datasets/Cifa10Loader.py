import numpy as np



class Cifa10Loader:
    def __init__(self):
        pass
    
    # load file
    def unpickle(self, path):
        import pickle
        import sys
        
        #python 2.7
        if sys.version_info[0] < 3:
            reload(sys)  # Reload does the trick!
            sys.setdefaultencoding('UTF8')
            with open(path, 'rb') as fo:
                dictionary = pickle.load(fo)
                
        else: #python 3.5+
            with open(path, 'rb') as fo:
                dictionary = pickle.load(fo, encoding='bytes')
        return dictionary

    # load each num_of_batch
    def load_cifa10_batch(self, path, num_of_batch):
        dataset = self.unpickle(path + num_of_batch)
        # weird key names: I found it out by printing dataset 
        x = np.array(dataset[b'data']);
        y = np.array(dataset[b'labels']);
    
        return x, y;
    
    def load_cifar10(self, path, numofbatch=1):
        xs = [];
        ys = [];
    
        for i in range(numofbatch):
            X, Y = self.load_cifa10_batch(path, 'data_batch_' + str(i + 1));
            xs.append(X);
            ys.append(Y);
        
        Xtr = np.concatenate(xs);
        Ytr = np.concatenate(ys);
        
        del X, Y;
    
        Xte, Yte = self.load_cifa10_batch(path, "test_batch");
    
        return Xtr, Ytr, Xte, Yte;
    

# print(test[b'labels']);
# print(test[b'data']);
