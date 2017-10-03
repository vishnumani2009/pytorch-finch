#import basic requirements
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as utils
import numpy as np
from sklearn.metrics import classification_report

dtype= torch.cuda.FloatTensor

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

class MNISTNET(nn.Module):
    def __init__(self):
        super(MNISTNET, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 120)
        self.fc1_drop = nn.Dropout()
        self.fc2 = nn.Linear(120, 10)
        self.epochs=10
       
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1_drop(self.fc1(x)))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x)
    
    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    def fit(self,traindataloader):
        self.optimizer=optim.SGD(self.parameters(),lr=0.01)
        self.criterion=torch.nn.MSELoss()
        self.train()
        for batch,(data,target) in enumerate(traindataloader):
            #data, target = data.cuda(), target.cuda()
            data,target=Variable(data),Variable(target)
            self.optimizer.zero_grad()
            #print data.size()
            output=self(data)
            loss=self.criterion(output,target)
            loss.backward()
            self.optimizer.step()
            if batch % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    1, batch * len(data), len(traindataloader.dataset),
                    100. * batch / len(traindataloader), loss.data[0]))

        
    def predict(self,testdataloader):
        self.eval()
        correct=0
        testloss=0
    
    #todo: must optimize bunch of functions
        for data,target in testdataloader:
            #data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = self(data).cpu().data.numpy()[0]
            output=np.array(output)
            op=[]
            for i in output:
                if i==np.max(output):
                    op.append(1)
                else:
                    op.append(0)
            ypred.append(op)
        ynewpr=[np.argmax(i,axis=-1) for i in ypred]
        ynewte=[np.argmax(i,axis=-1) for i in y_test]
        print(classification_report(ynewte, ynewpr))