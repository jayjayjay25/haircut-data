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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm
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
        #same network as in ml_stats_final_cnn.py once parameters are reloaded, except this time functions are ordered according to how the
        #model processes them.
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=(3,3),stride=1,padding=1)
        self.batch1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU()
        self.max1 = nn.MaxPool2d((2,2))

        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),stride=1,padding=1)
        self.batch2 = nn.BatchNorm2d(64)
        self.act2 = nn.ReLU()
        self.max2 = nn.MaxPool2d((2,2))

        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),stride=1,padding=1)
        self.batch3 = nn.BatchNorm2d(128)
        self.act3 = nn.ReLU()
        self.max3 = nn.MaxPool2d((2,2))
        
        self.conv4 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(3,3),stride = 1, padding=1)
        self.batch4 = nn.BatchNorm2d(256)
        self.act4 = nn.ReLU()
        self.max4 = nn.MaxPool2d((2,2))
        
        self.conv5 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(3,3), stride = 1, padding = 1)
        self.batch5 = nn.BatchNorm2d(512)
        self.act5 = nn.ReLU()
        self.max5 = nn.MaxPool2d((2,2))
        
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(512*9*9, 1024)
        self.act6 = nn.ReLU()
        self.drop_it  = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024,22)

    def forward(self, input):
        c1 = self.conv1(input)
        b1 = self.batch1(c1)
        r1 = F.relu(b1)
        #b1 = self.batch1(c1)
        s2 = F.max_pool2d(r1, (2, 2))

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

#I don't use this function, but its been copied over from the other file mentioned just in case it
#should be used here, on the smaller dataset
def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X/255
            #X = X.to(device)
            #y = y.to(device)
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

# torch.manual_seed(123)
# # device = (
# #     "cuda"
# #     if torch.cuda.is_available()
# #     else "cpu"
# # )
# # print(f"Using {device} device")    
# model = Net().apply(init_weights)
# #model = CNN_smaller().apply(init_weights)
# # model = model.to(device)

# path = "haircut_data"
# transform = T.Compose([T.Resize((300, 300)),T.ToTensor()])
# dataset = ImageFolder(root=path, transform=transform)
# rough_dataset = ImageFolder(root = path)
# learning_rate = 0.001

# train_length=int(0.7* len(dataset))
# test_length=len(dataset)-train_length

# train_dataset,test_dataset=torch.utils.data.random_split(dataset,(train_length,test_length))

# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# checkpoint = torch.load('cnn_final_v4_model.pt')
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# #print(list(model.children()))
# feature_extractor = torch.nn.Sequential(*list(model.children())[:-2])
# print(list(model.children())[:-2])
# #def feature_extractor(input):
# #    model(input)

# train_dataloader =DataLoader(train_dataset, batch_size=1, shuffle=True)
# test_dataloader = DataLoader(test_dataset, shuffle=False,batch_size=1)
# #X,y = train_dataset[0]
# #X = X/255
# #X = X.to(device)
# #output = feature_extractor(X)
# #print(X)

# feat_X = list()
# y_train = list()
# test_X = list()
# y_test = list()
# with torch.no_grad():
#     for (X,y) in train_dataloader:
#         X = X/255
#         #X = X.to(device)
#         outs = feature_extractor(X)
#         feat_X.append(np.array(outs).squeeze())
#         y_train.append(np.array(y)[0])
#     for (X,y) in test_dataloader:
#         X = X/255
#         #X = X.to(device)
#         outs = feature_extractor(X)
#         test_X.append(np.array(outs).squeeze())
#         y_test.append(np.array(y)[0])
# #print(y_train)
# feat_X  = pd.DataFrame(feat_X)
# test_X = pd.DataFrame(test_X)

# fn = 'cnn_features_train.csv'
# filepath = Path(fn)
# filepath.parent.mkdir(parents=True, exist_ok=True)
# data_train = pd.DataFrame(np.column_stack((feat_X, y_train)))
# data_train.to_csv(filepath, index=False) 

# fn = 'cnn_features_test.csv'
# filepath = Path(fn)
# filepath.parent.mkdir(parents=True, exist_ok=True)
# data_test = pd.DataFrame(np.column_stack((test_X, y_test)))
# data_test.to_csv(filepath, index=False) 

#Same as the file in cnn_feature_extract, except this file uses those extracted features directly in
#SVM from a presaved file, so that we don't have to keep extracting them from the large CNN every time.
with open("cnn_features_train.csv", mode ='r') as file:
        df = pd.read_csv(file)
#print(df)
df.columns = df.columns.map(str)
feat_X = df.drop(columns = "1024")
#print(df)
y_train = df["1024"]

with open("cnn_features_test.csv", mode ='r') as file:
        df2 = pd.read_csv(file)
df2.columns = df2.columns.map(str)
test_X = df2.drop(columns = "1024")
y_test = df2["1024"]

clf = make_pipeline(StandardScaler(), svm.SVC())
clf.fit(feat_X,y_train)
preds_train = clf.predict(feat_X)
#print(preds)
mspe_train = sum((preds_train - y_train)**2 )/len(y_train)
accuracy = sum((preds_train == y_train))/len(y_train)
print(mspe_train)
print(accuracy)

preds_train = clf.predict(test_X)
#print(preds)
mspe_test = sum((preds_train - y_test)**2 )/len(y_test)
accuracy = sum((preds_train == y_test))/len(y_test)
print(mspe_test)
print(accuracy)



#train_dataloader =DataLoader(train_dataset, batch_size=1, shuffle=True)
#test_dataloader = DataLoader(test_dataset, shuffle=False,batch_size=1)

