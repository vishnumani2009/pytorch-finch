#import basic requirements
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as utils
import numpy as np
from mnist_mlp import MNISTNET
#dtype= torch.cuda.FloatTensor

def load_data(path='mnist.npz'):
    """Loads the MNIST dataset.
    # Arguments
        path: path where mnist is stored locallym, download from https://s3.amazonaws.com/img-datasets/mnist.npz
            
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)

def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]

    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical


def createdata():
    batch_size = 128
    num_classes = 10
    epochs = 20

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = load_data()

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    #print(x_train.shape, 'train samples')
    #print(x_test.shape, 'test samples')
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
    y_train=to_categorical(y_train,10)
    y_test=to_categorical(y_test,10)
    ypred=[]
    
    return x_train,y_train,x_test,y_test


def main():
    xtr,ytr,xte,yte=createdata()
    net=MNISTNET()
    #net=net.cuda()
    #convert numpy arrays to torch tensors
    Xtr=torch.stack([torch.Tensor(i) for i in xtr])
    Ytr=torch.stack([torch.Tensor(i) for i in ytr])

    Xte=torch.stack([torch.Tensor(i) for i in xte])
    Yte=torch.stack([torch.Tensor(i) for i in yte])

    traindataset=utils.TensorDataset(Xtr,Ytr)
    traindataloader=utils.DataLoader(traindataset)


    testdataset=utils.TensorDataset(Xte,Yte)
    testdataloader=utils.DataLoader(testdataset)


    #define hyperparameters
    
    for i in range(1):
        net.fit(traindataloader)
        net.predict(testdataloader)

main()