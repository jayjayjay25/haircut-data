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

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        #self.flatten = nn.Flatten()

        self.conv_relu_encode = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=(3,3),stride=1,padding=1),
            #nn.Sigmoid(),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),stride=1,padding=1),
            #nn.Sigmoid(),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),stride=1,padding=1),
            #nn.Sigmoid(),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            #nn.Dropout(0.25),

            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(3,3),stride = 1, padding=1),
            #nn.Sigmoid(),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),

            
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(3,3), stride = 1, padding = 1),
            #nn.Sigmoid(),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),

            nn.Flatten(),
            nn.Linear(512*9*9, 1024),
            #nn.Sigmoid(),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024,22),
            #nn.Sigmoid(),
            #nn.ReLU()
            #nn.Softmax(),
        )

    def forward(self, x):
        #x = self.Flatten(x)
        logits = self.conv_relu_encode(x)
        return logits


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
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
        # an affine operation: y = Wx + b
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

class Net_smaller(nn.Module):

    def __init__(self):
        super(Net_smaller, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=8,kernel_size=(3,3),stride=1,padding=1)
        self.batch1 = nn.BatchNorm2d(8)
        self.act1 = nn.ReLU()
        self.max1 = nn.MaxPool2d((2,2))

        self.conv2 = nn.Conv2d(in_channels=8,out_channels=16,kernel_size=(3,3),stride=1,padding=1)
        self.batch2 = nn.BatchNorm2d(16)
        self.act2 = nn.ReLU()
        self.max2 = nn.MaxPool2d((2,2))

        self.conv3 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),stride=1,padding=1)
        self.batch3 = nn.BatchNorm2d(32)
        self.act3 = nn.ReLU()
        self.max3 = nn.MaxPool2d((2,2))
        
        self.fc1 = nn.Linear(32*37*37, 1024)
        self.drop_it  = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024,22)
        #F.batch_norm

    def forward(self, input):
        c1 = self.conv1(input)
        b1 = self.batch1(c1)
        r1 = self.act1(b1) 
        m1 = self.max1(r1)

        c2 = self.conv2(m1)
        b2 = self.batch2(c2)
        r2 = self.act2(b2) 
        m2 = self.max2(r2)

        c3 = self.conv3(m2)
        b3 = self.batch3(c3)
        r3 = self.act3(b3) 
        m3 = self.max3(r3)
        #print(m3.shape)
        fl = torch.flatten(m3,1)
        f1 = F.relu(self.fc1(fl))
        f2 = self.drop_it(f1)
        output = self.fc2(f2)
        return output



def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
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
        #l1_norm = sum(p.abs().sum() for p in model.parameters())
        #loss = loss + 0.01 * l1_norm
        l2_norm = sum(p.pow(2).sum() for p in model.parameters())
        loss = loss + 0.01 * l2_norm
        # Backpropagation
        loss.backward()
        optimizer.step()
        train_loss += loss_fn(pred, y).item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
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
    #plt.plot(loss_list,batch_num)
    #plt.show()
    return loss, loss_list
        
        
def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
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

torch.manual_seed(123)
#loading data
path = "haircut_more_data"
transform = T.Compose([T.Resize((300, 300)),T.ToTensor()])

#rough_dataset = ImageFolder(root = path)



dataset = ImageFolder(root=path, transform=transform)
#img, label = rough_dataset[387]
#print(img.size)
#small_img = img.resize((300,300))
#print(len(rough_dataset))
#plt.title(label)
#plt.axis("off")
#plt.imshow(small_img)
#plt.show()


#dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using {device} device")    
#model = CNN().apply(init_weights)
#model = CNN_smaller().apply(init_weights)
model = Net_smaller().apply(init_weights)
model = model.to(device)

#X = torch.rand(3,32,32,device=device)
#logits = model(X)

learning_rate = 0.001

train_length=int(0.7* len(dataset))
test_length=len(dataset)-train_length

train_dataset,test_dataset=torch.utils.data.random_split(dataset,(train_length,test_length))
train_dataloader =DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, shuffle=False,batch_size=8)

# for batch_number, (images, labels) in enumerate(train_dataloader):
#     print(batch_number, labels)
#     #print(images)
#     for i in range(len(images)):
#         images[i].imshow()
#         break
#     break

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#checkpoint = torch.load('cnn_final_v4_model.pt')
#model.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

loss_lists = list()
epochs = 50
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    tic = time.time()
    loss, loss_list = train_loop(train_dataloader, model, loss_fn, optimizer)
    #print(loss_list.shape)
    #print(loss_lists.shape)
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
        }, 'cnn_final_v3_model.pt')
    if t % 51 == 0 and t != 0:
        plt.plot(range(0,len(loss_lists)), loss_lists)
        plt.show()


#Get some numpies for the svm to be used for scikit learn
#my_nparray = my_tensor.numpy()