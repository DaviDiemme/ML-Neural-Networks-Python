'''
Neural Network Model by Davide De Marco

uses pytorch to train a simple deep learning model to process grayscale images
from the fashionMNIST dataset and classify clothes divided in 10 classes
'''

''' Import the fundamental modules '''
import sys
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from IPython import display
from torch import nn
from torch.utils import data
from torchvision import transforms

argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)

"""Some class and function definitions"""

''' Define accumulator class for accumulating sums over `n` variables '''
class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


""" Plotting data in animation """
class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        
        # Incrementally plot multiple lines
        if legend is None:
            legend = []

        # Use the svg format to display a plot in Jupyter
        #display.set_matplotlib_formats('svg')
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
            
        # Use a lambda function to capture arguments
        self.config_axes = lambda: set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    
    # Add multiple data points into the figure
    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


""" Set the axes for matplotlib """
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


''' Weight instantiation and initialisation '''
def init_weights(m):
    # check type to init different layers in different ways
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, std=0.01)
        torch.nn.init.zeros_(m.bias)


''' Compute the number of correct predictions '''
def accuracy(y_hat, y):  #y_hat is a matrix; 2nd dimension stores prediction scores for each class.

    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)        # Predicted class is the index of max score    
        
    cmp = (y_hat.type(y.dtype) == y)        # because`==` is sensitive to data types
    return float(torch.sum(cmp))



''' Compute the accuracy for a model on a dataset '''
def eval_accuracy(net, data_iter):
    
    # Set the model to evaluation mode
    if isinstance(net, torch.nn.Module):
        net.eval()
        
    metric = Accumulator(2)  # No. of correct predictions, no. of predictions
    for _, (X, y) in enumerate(data_iter):
        metric.add(accuracy(net(X), y), y.numel())
        
    return metric[0] / metric[1]


''' Train model and compute accuracy for a single epoch '''
def train_epoch(net, train_iter, loss, updater):
# Set the model to training mode
    if isinstance(net, torch.nn.Module):
        net.train()
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = Accumulator(3)
    
    for X, y in train_iter:
        # Compute gradients and update parameters
        y_hat = net(X)
        l = loss(y_hat, y)                                                          # calculate loss
        if isinstance(updater, torch.optim.Optimizer):                              # if updater is a class torch.optim.Optimizer
            updater.zero_grad()                                                     # calculate gradients with optimizer 
            l.backward()                                                            # backpropagation
            updater.step()                                                          # optimizer step ahead
            metric.add(float(l) * len(y), accuracy(y_hat, y), y.size().numel())     # add to accumulator loss, no. of correct predictions and no. of predictions
        else:
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
            
    return metric[0] / metric[2], metric[1] / metric[2]                             # Return training loss and training accuracy


''' Train a model '''
def train(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(xlabel='epoch', xlim=[0, num_epochs], ylim=[0, 1], legend=['train loss', 'train acc', 'test acc'])
    
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_iter, loss, updater)
        test_acc = eval_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
        
    train_loss, train_acc = train_metrics
    print('train final accuracy: ', train_acc)
    print('test final loss: ', train_loss)
    print('test final accuracy: ', test_acc)




''' Create Neural Network Model '''

class Net(torch.nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super(Net, self).__init__()

        self.num_hidden = num_hidden
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.backbone_mlp1 = nn.Sequential(
                          nn.Linear(num_hidden, num_hidden), 
                          nn.ReLU(),
                          nn.Linear(num_hidden, num_hidden))
      
        self.backbone_mlp2 = nn.Sequential(
                          nn.Linear(num_inputs, num_inputs), 
                          nn.ReLU(), 
                          nn.Linear(num_inputs, num_inputs))

    def forward(self, x):

        x = self.Stem(x, 7)
      
        x = torch.transpose(x, 1, 2)    # transpose before first backbone mlp
        x = self.backbone_mlp1(x)

        x = torch.transpose(x, 1, 2)  # transpose before second backbone mlp
        x = self.backbone_mlp2(x)
      
        out = self.Classifier(x)

        return out
    
    def Stem(self, x, PATCH_SIZE):
        # A function to split the input image into square patches
        # and then rearrange the pixels in each patch.
        # The input image of size (h, w) is rearranged into (num_patches, patch_size^2).
      
        # - PATCH_SIZE is the lenght of the square patch in pixels
        # (i.e. 7x7 patch: PATCH_SIZE=7).

        # - num_patches is the number of patches in the image,
        # it is calculated by dividing the image size by the patch size
        
        PATCH_SIZE
        num_patches = (28//PATCH_SIZE)**2
        unfold = nn.Unfold(kernel_size=(7,7), stride=(7,7))

        x_divided = x.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
        
        x_divided = x_divided.reshape( -1, num_patches, PATCH_SIZE, PATCH_SIZE)
              
        return x_divided.reshape(-1, num_patches, PATCH_SIZE**2)

    def Classifier(self, x):
        # A function to extract the mean feature vector
        # from the feature matrix and that will be directly used for the
        # classification and training process
        
        x = x.mean(1, False)
        #print('after mean: ', x.size())      

        return x
  
  

''' Download the Fashion-MNIST dataset, load it into memory, and read training and test batches '''
resize = None
batch_size = 256
trans = [transforms.ToTensor()]

if resize:
    trans.insert(0, transforms.Resize(resize))

trans = transforms.Compose(trans)
mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4)
test_iter = data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4) # num_workers:processes to read the data ( use 4)


num_inputs, num_hidden, num_outputs = 49, 16, 10
net = Net(num_inputs, num_hidden, num_outputs)
net.apply(init_weights)

# Create loss here. Use Cross Entropy loss:
loss = nn.CrossEntropyLoss()

lr = 0.6
wd = 0.0002

# Create optimizer here. Use SGD with weight decay wd and learning rate lr.
optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)

num_epochs = 30
train(net, train_iter, test_iter, loss, num_epochs, optimizer)