'''
Created on Jun 19, 2017

@author: demon
'''
from caffe2.python import core, utils,workspace, visualize, net_drawer,brew,model_helper
#from caffe2.python.visualize import NCHW
from caffe2.proto import caffe2_pb2
import numpy as np
from matplotlib import pyplot
from com.chandler.irac.datasets.Cifa10Loader import Cifa10Loader
import os
import math

# display image from np arrays using matplotlib
def visulize_batch(images,max_images,labels):
    pyplot.figure()
    length = images.shape[0]
    dim = int(math.ceil(math.sqrt(length)))
    dim = min(dim,max_images)
    
    for i in range(dim**2):
        pyplot.subplot(dim,dim,i+1)
        pyplot.imshow(images[i])
        #pyplot.axis('on')
        pyplot.title('Class index: '+str(labels[i]))
        
    pyplot.show()


#shuffle input and labels
def shuffle(data, labels):
    if(data.shape[0] != labels.shape[0]):
        print("Cannot shuffle two datasets with different length! System exit!")
        import sys
        sys.exit()
    
    random_index = np.random.permutation(data.shape[0])
    data = data[random_index]
    labels = labels[random_index]
    return data,labels




def save_cifar10_to_db(db_name_train,db_name_test):
    if not (os.path.exists(db_name_train) or os.path.exists(db_name_test)):
        loader = Cifa10Loader();
        x_train, y_train, x_test, y_test = loader.load_cifar10('../../../../../data/CIFAR10/', 5)
        x_train = x_train.astype('float32')
        y_train = y_train.astype(int)
        x_test = x_test.astype('float32')
        y_test = y_test.astype(int)
        #reshape data
        x_train = x_train.reshape(x_train.shape[0], 3, 32, 32)
        x_test = x_test.reshape(x_test.shape[0], 3, 32, 32)
        
        '''
        #caffe2 show image
        NCHW.ShowSingle(x_train[0])
        '''
                
        '''
        #swap axis manually
        x_train = np.swapaxes(x_train, 1, 2)
        x_train = np.swapaxes(x_train, 2, 3)
        x_test = np.swapaxes(x_test, 1, 2)
        x_test = np.swapaxes(x_test, 2, 3)
        '''
        x_train = visualize.ChannelLast(x_train)
        x_test = visualize.ChannelLast(x_test)
        print('x_train shape:', x_train.shape)
        print('y_train shape:', y_train.shape)        
        
        visulize_batch(x_train,8, y_train)
        insert_db(db_name_train,x_train,y_train)
        insert_db(db_name_test,x_test,y_test)



def insert_db(db_name,data,labels):    
    db_type = db_name.split(".")[-1]
    db = core.C.create_db(db_type, db_name, core.C.Mode.write)
    transaction = db.new_transaction()
    for i in range(data.shape[0]):
        data_and_label = caffe2_pb2.TensorProtos()
        data_and_label.protos.extend([
            utils.NumpyArrayToCaffe2Tensor(data[i]),
            utils.NumpyArrayToCaffe2Tensor(labels[i])])
        transaction.put(
            'data_%03d'.format(i),
            data_and_label.SerializeToString())
    # Close the transaction, and then close the db.
    del transaction
    del db


#only the first two loops are correct then keeps getting the first batch?????????????????
def load_data(model,db_name,batch_size):
    net_proto = model.net
    db_type = db_name.split(".")[-1]
    dbreader = net_proto.CreateDB([], "dbreader", db=db_name_train, db_type=db_type)
    data, labels = net_proto.TensorProtosDBInput([dbreader], ["data", "labels"], batch_size=batch_size)
    
    # scale data from [0,255] down to [0,1]
    data = train_model.Scale(data, data, scale=float(1./256))
    # don't need the gradient for the backward pass
    data = model.StopGradient(data, data)
    return data,labels


def load_data2(model,db_name,batch_size):
    db_type = db_name.split(".")[-1]
    # load the data
    data, labels = model.TensorProtosDBInput(
        [], ["data", "labels"], batch_size=batch_size,
        db=db_name, db_type=db_type)
    # cast the data to float
    #data = model.Cast(data, "data", to=core.DataType.FLOAT)
    #labels = model.Cast(labels, "labels", to=core.DataType.INT)
    
    # scale data from [0,255] down to [0,1]
    data = train_model.Scale(data, data, scale=float(1./256))
    # don't need the gradient for the backward pass
    data = model.StopGradient(data, data)
    return data,labels


def plot_mini_net(file_name,net_proto):
    #plot the net
    graph = net_drawer.GetPydotGraphMinimal(net_proto.Proto().op, rankdir="LR", minimal_dependency=True)
    graph.write_png(file_name)
    
def plot_net(file_name,net_proto):
    #plot the net
    graph = net_drawer.GetPydotGraph(net_proto.Proto().op, net_proto.Proto().name, rankdir="LR")
    graph.write_png(file_name) 
    


def add_layers(model, data):
    # Image size: 32 x 32 x 3 -> 30 x 30 x 60 
    conv1 = brew.conv(model, data, 'conv1', dim_in=3, dim_out=64, kernel=3)
    #add relu
    conv1 = brew.relu(model,conv1,conv1)
    # Image size: 30 x 30 -> 15 x 15
    pool1 = brew.max_pool(model, conv1, 'pool1', kernel=2, stride=2)
    # Image size: 15 x 15 -> 12 x 12
    conv2 = brew.conv(model, pool1, 'conv2', dim_in=64, dim_out=128, kernel=3)
    #add relu
    conv2 = brew.relu(model,conv2,conv2)
    # Image size: 12 x 12 -> 6 x 6
    pool2 = brew.max_pool(model, conv2, 'pool2', kernel=2, stride=2)
    # 32 * 6 * 6 stands for dim_out from previous layer multiplied by the image size
    # Hidden layer
    fc3 = brew.fc(model, pool2, 'fc3', dim_in=128 * 6 * 6, dim_out=512)
    #add relu
    fc3 = brew.relu(model, fc3, fc3)
    #CO = model.param_init_net.ConstantFill([], "CO", shape=[1], value=0.18)
    #fc3 = brew.prelu(model, fc3, fc3)
    #output layer
    pred = brew.fc(model, fc3, 'pred', 512, 10)    
    #add softmax
    probs = brew.softmax(model, pred, 'probs')
    return probs

def add_accuracy(model, probs, label):
    accuracy = brew.accuracy(model, [probs, label], "accuracy")
    return accuracy

def add_loss_and_backpropagation(model, probs, labels):
    # define cross-entropy
    x_entropy = model.LabelCrossEntropy([probs, labels], 'x_entropy')
    # compute the expected loss
    loss = model.AveragedLoss(x_entropy, "loss")
    # use the average loss we just computed to add gradient operators to the model
    model.AddGradientOperators([loss])
    # a counter
    ITER = brew.iter(model, "ITER")
    # set the learning rate schedule, stepsize is iteration size, gamma is simply lr * gamma
    LR = model.LearningRate(
        ITER, "LR", base_lr=-0.01, policy="step", stepsize=4, gamma=0.999)
    #regularization
    #brew.add_weight_decay(model, 0.001)
    # ONE is a constant value that is used in the gradient update. We only need
    # to create it once, so it is explicitly placed in param_init_net.
    ONE = model.param_init_net.ConstantFill([], "ONE", shape=[1], value=1.0)
    # Now, for each parameter, we do the gradient updates.
    for param in model.params:
        # Note how we get the gradient of each parameter - ModelHelper keeps
        # track of that.
        param_grad = model.param_to_grad[param]
        # The update is a simple weighted sum: param = param + param_grad * LR
        model.WeightedSum([param, ONE, param_grad, LR], param)
    '''
    # let's checkpoint every 20 iterations, which should probably be fine.
    model.Checkpoint([ITER] + model.params, [],
                   db="cifar10_checkpoint_%05d.minidb",
                   db_type="minidb", every=20)
                   '''
    
    
def add_tracking(model):
    # to_file=1 routes the printed output to a file otherwise to stdout. 
    # The file is going to be stored under: root_folder/[blob name]
    model.Print('accuracy', [], to_file=0)
    model.Print('loss', [], to_file=0)
    # Summarizes the parameters. Different from Print, Summarize gives some
    # statistics of the parameter, such as mean, std, min and max.
    '''
    for param in model.params:
        model.Summarize(param, [], to_file=1)
        model.Summarize(model.param_to_grad[param], [], to_file=1)
        '''





'''
fig, ax = pyplot.subplots()

x = np.arange(0, 2*np.pi, 0.01)        # x-array
line, = ax.plot(x, np.sin(x))

def animate(i):
    line.set_ydata(np.random.rand(1))  # update the data
    return line,

#Init only required for blitting to give a clean slate.
def init():
    line.set_ydata(np.ma.array(x, mask=True))
    return line,

ani = animation.FuncAnimation(fig, animate, init_func=init,
    interval=25, blit=True)
pyplot.show()
'''






#save to minidb if not exist
db_name_train = "cifar10_train.minidb"
db_name_test = "cifar10_test.minidb"
save_cifar10_to_db(db_name_train,db_name_test)








#init training model helper
arg_scope = {"order": "NHWC"}
train_model = model_helper.ModelHelper(name="cifar10_train", arg_scope=arg_scope)
#set to use GPU
train_model.param_init_net.RunAllOnGPU()
train_model.net.RunAllOnGPU()

#Note: if we call RunNet multiple times, it fetches one batch after another
#will not keep fetching the same data
batch_size = 64
x_train,y_train = load_data2(train_model, db_name_train, batch_size)
# Neural network model
probs = add_layers(train_model, x_train)
# track the accuracy of the model
add_accuracy(train_model, probs, y_train)
add_loss_and_backpropagation(train_model,probs,y_train)
add_tracking(train_model)

#plot net
plot_net("graph.png", train_model.net)




#init params (W,b etc.)
workspace.RunNetOnce(train_model.param_init_net)
# creating the network, must call create before RunNet
# Note that RunNetOnce can be called without CreateNet
workspace.CreateNet(train_model.net)
# set the number of iterations from batch size 781 = round(50000/64)
# 78100 means 100 epochs
total_iters = 24000
accuracy = np.zeros(total_iters)
loss = np.zeros(total_iters)
# Now, we will manually run the network for 200 iterations.
for i in range(total_iters):
    print("Loop------------>"+str(i)+"  "+str(workspace.FetchBlob("LR")))
    workspace.RunNet(train_model.net)
    accuracy[i] = workspace.FetchBlob('accuracy')
    loss[i] = workspace.FetchBlob('loss')
    
    '''
    if(i%2000==0):
        pyplot.plot(loss, 'b')
        pyplot.plot(accuracy, 'r')
        pyplot.legend(('Loss', 'Accuracy'), loc='upper right')
        
        pyplot.show()
    
    
    #debug--------------------------------------
    visulize_batch(workspace.FetchBlob("data"),4,workspace.FetchBlob("labels"))
    if(i==50):
        print(np.argmax(workspace.FetchBlob("probs")[:16],axis=1))
        visulize_batch(workspace.FetchBlob("data"), 4, workspace.FetchBlob("labels"))
   
#debug--------------------------------------
print(np.argmax(workspace.FetchBlob("probs")[:16],axis=1))
visulize_batch(workspace.FetchBlob("data"), 4, workspace.FetchBlob("labels"))
'''
# After the execution is done, let's plot the values.
pyplot.plot(loss, 'b')
pyplot.plot(accuracy, 'r')
pyplot.legend(('Loss', 'Accuracy'), loc='upper right')

pyplot.show()






# Testing model. We will set the batch size to 100, so that the testing
# pass is 100 iterations (10,000 images in total).
test_model = model_helper.ModelHelper(
    name="cifar10_test", arg_scope=arg_scope, init_params=False)
batch_size_test = 100
x_test,y_test = load_data2(test_model, db_name_test, batch_size_test)
probs = add_layers(test_model, x_test)
add_accuracy(test_model, probs, y_test)

test_model.param_init_net.RunAllOnGPU()
test_model.net.RunAllOnGPU()
#using the trained params, but still needs to call this
workspace.RunNetOnce(test_model.param_init_net)
# run a test pass on the test net
workspace.CreateNet(test_model.net, overwrite=True)
test_accuracy = np.zeros(100)
for i in range(100):
    workspace.RunNet(test_model.net.Proto().name)
    test_accuracy[i] = workspace.FetchBlob('accuracy')
    '''
    #debug--------------------------------------
    if(i==50):
        print(np.argmax(workspace.FetchBlob("probs")[:16],axis=1))
        visulize_batch(workspace.FetchBlob("data"), 4, workspace.FetchBlob("labels"))

#debug--------------------------------------
print(np.argmax(workspace.FetchBlob("probs")[:16],axis=1))
visulize_batch(workspace.FetchBlob("data"), 4, workspace.FetchBlob("labels"))
'''
# After the execution is done, let's plot the values.
pyplot.plot(test_accuracy, 'r')
pyplot.title('Acuracy over test batches.')
print('test_accuracy: %f' % test_accuracy.mean())
pyplot.show()







'''
# Deployment model. We simply need the main LeNetModel part.
deploy_model = model_helper.ModelHelper(
    name="cifar10_deploy", arg_scope=arg_scope, init_params=False)
add_layers(deploy_model, y_test)
# You may wonder what happens with the param_init_net part of the deploy_model.
# No, we will not use them, since during deployment time we will not randomly
# initialize the parameters, but load the parameters from the db.
'''










'''
workspace.RunNetOnce(train_model.param_init_net)
workspace.CreateNet(train_model.net)
workspace.RunNet(train_model.net)
#print(train_model.net.Proto())
print("x_train shape is: "+format(workspace.FetchBlob("data").shape))
#print("x_train sample data: "+format(workspace.FetchBlob("data")[:2]))
print("y_train shape is: "+format(workspace.FetchBlob("labels").shape))
print("y_train sample data: "+format(workspace.FetchBlob("labels")[:80]))
#visulize_batch(workspace.FetchBlob("data"),8,workspace.FetchBlob("labels"))

workspace.RunNet(train_model.net)
print("x_train shape is: "+format(workspace.FetchBlob("data").shape))
#print("x_train sample data: "+format(workspace.FetchBlob("data")[:2]))
print("y_train shape is: "+format(workspace.FetchBlob("labels").shape))
print("y_train sample data: "+format(workspace.FetchBlob("labels")[:80]))
#visulize_batch(workspace.FetchBlob("data"),8,workspace.FetchBlob("labels"))

workspace.RunNet(train_model.net)
print("x_train shape is: "+format(workspace.FetchBlob("data").shape))
#print("x_train sample data: "+format(workspace.FetchBlob("data")[:2]))
print("y_train shape is: "+format(workspace.FetchBlob("labels").shape))
print("y_train sample data: "+format(workspace.FetchBlob("labels")[:80]))
#visulize_batch(workspace.FetchBlob("data"),8,workspace.FetchBlob("labels"))
'''




