import numpy as np
from numpy import random as ran
import pandas as pd
from pathlib import Path
import random
import time
import math
import torch
from PIL import Image
import sklearn.cluster as c
#from sklearn.cluster import Kmeans
import os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import torch.nn.functional as F


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        #The main CNN I used
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=(3,3),stride=1,padding=1)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),stride=1,padding=1)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),stride=1,padding=1)
        self.conv4 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(3,3),stride = 1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(3,3), stride = 1, padding = 1)
        self.batch1 = nn.BatchNorm2d(32)
        self.batch2 = nn.BatchNorm2d(64)
        self.batch3 = nn.BatchNorm2d(128)
        self.batch4 = nn.BatchNorm2d(256)
        self.batch5 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(512*9*9, 1024)
        self.drop_it  = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024,22)
        #F.batch_norm

    def forward(self, input):
        c1 = F.relu(self.batch1(self.conv1(input)))
        #b1 = self.batch1(c1)
        s2 = F.max_pool2d(c1, (2, 2))

        c3 = F.relu(self.batch2(self.conv2(s2)))
        #b2 = self.batch2(c3)
        s4 = F.max_pool2d(c3, (2,2))

        c5 = F.relu(self.batch3(self.conv3(s4)))
        #b3 = self.batch3(c5)
        s6 = F.max_pool2d(c5, (2,2))

        c7 = F.relu(self.batch4(self.conv4(s6)))
        #b4 = self.batch4(c7)
        s8 = F.max_pool2d(c7, (2,2))

        c9 = F.relu(self.batch5(self.conv5(s8)))
        #b5 = self.batch5(c9)
        s10 = F.max_pool2d(c9,(2,2))

        s11 = torch.flatten(s10,1)
        f12 = F.relu(self.fc1(s11))
        f13 = self.drop_it(f12)
        output = self.fc2(f13)
        return output


#The training loop used to train the model and get training errors
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    tic = time.time()
    loss_list = []
    batch_num = []
    train_loss, correct = 0, 0
    num_batches = len(dataloader)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X/255
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        optimizer.zero_grad()
        loss = loss_fn(pred, y).to(device)
        #Regularization
        l2_norm = sum(p.pow(2).sum() for p in model.parameters())
        loss = loss + 0.01 * l2_norm
        # Backpropagation
        loss.backward()
        optimizer.step()
        train_loss += loss_fn(pred, y).item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        #Some code to be uncommented if outputs within the epoch are so desired
        if batch % 25 == 0:
            #guesses = [torch.argmax(i) for i in pred]
            #print(guesses)
            #print(y)
            loss, current = loss.item(), (batch+1)*len(X)
            loss_list.append(loss)
            #batch_num.append(batch)
            #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            #toc = time.time()
            #print(f"Time Elapsed: {toc-tic}")
    train_loss /= num_batches
    correct /= size
    print(f"Train Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {train_loss:>8f} \n")
    return loss, loss_list
        
#The test loop used to get test errors
def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X/255
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform(m.weight)
        m.bias.data.fill_(0.01)


########## MAIN ##############

torch.manual_seed(123) #used a seed for both reproducibility of the model and the dataset train/test split
#loading data
path = "haircut_more_data" #7585 images, 22 labels
transform = T.Compose([T.Resize((300, 300)),T.ToTensor()])

#rough_dataset = ImageFolder(root = path)
dataset = ImageFolder(root=path, transform=transform)


#this model uses my GPU to run. Not extremely faster by any means but increased processing by a couple seconds at a time goes a long way.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using {device} device")
model = Net().apply(init_weights)
model = model.to(device)


learning_rate = 0.001

train_length=int(0.7* len(dataset))
test_length=len(dataset)-train_length

train_dataset,test_dataset=torch.utils.data.random_split(dataset,(train_length,test_length)) #random 70/30 split
train_dataloader =DataLoader(train_dataset, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_dataset, shuffle=False,batch_size=1)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#Uncomment below three lines to use the weights of the network obtained from 30 epochs
#checkpoint = torch.load('cnn_final_v4_model.pt')
#model.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

loss_lists = list()
epochs = 50
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    tic = time.time()
    loss, loss_list = train_loop(train_dataloader, model, loss_fn, optimizer)
    loss_lists = loss_lists + loss_list
    toc = time.time()
    #print(f"Time Elapsed: {toc-tic}")
    tic = time.time()
    if t % 3 == 0:
        test_loop(test_dataloader, model, loss_fn)
    toc = time.time()
    #print(f"Time Elapsed: {toc-tic}")

    if t % 5 == 0 and t != 0 :
        torch.save({
        'epoch': t,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        }, 'cnn_final_v4_model.pt')
    if t % 51 == 0 and t != 0:
        plt.plot(range(0,len(loss_lists)), loss_lists) #plot the training loss change over time
        plt.show()


#Get some numpies for the svm to be used for scikit learn
#my_nparray = my_tensor.numpy()