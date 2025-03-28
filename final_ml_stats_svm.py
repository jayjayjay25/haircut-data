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
import os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

##Current function performs k=20-means clustering on each image in the dataset based on color alone
#If row and column are added to the image_df before clustering, the algorithm will also cluster
#based on pixel coordinates
def feature_extraction(dataset):
    k = 20
    X = []
    y = np.zeros(len(dataset))
    for i in range(len(dataset)):
        img, label = dataset[i]
        img  = img.resize((300,300))
        img_mat = np.array(img)
        dims = img_mat.shape
        row_indices, col_indices = np.indices((dims[0], dims[1]))
        img_mat = img_mat.reshape(-1, img_mat.shape[-1]) #/255 -- this rescales them, to get my first attempt
        image_df = pd.DataFrame(img_mat, columns= ["R","G","B"])
        kmeans = c.KMeans(n_clusters=k).fit(image_df)
        row_indices = row_indices.flatten()
        image_df["row"] = row_indices
        col_indices = col_indices.flatten()
        image_df["column"] = col_indices
        image_df["labels"] = kmeans.labels_

        areas = np.zeros(k)
        bbox = np.zeros((k,4))
        for j in range(len(areas)):
                areas[j] = len(image_df[image_df["labels"]==j])
                cluster = image_df[image_df["labels"]==j]
                seg_coords = np.array(cluster[["row","column"]])
                #print(seg_coords.shape)
                if len(seg_coords.shape) >0:
                        x_min = int(np.min(seg_coords[1]))
                        x_max = int(np.max(seg_coords[1]))
                        y_min = int(np.min(seg_coords[0]))
                        y_max = int(np.max(seg_coords[0]))
                        bbox[j] = [x_min, x_max, y_min, y_max]
                else:
                      bbox[j] = [0,0,0,0]
        #Note: add "x coordinates" and "y coordinates" before "Area" if chose to cluster based on coordinates
        naive_segments = pd.DataFrame(np.column_stack((kmeans.cluster_centers_, areas, bbox)), columns = ["R","G","B", "Area", "x_min", "x_max", "y_min", "y_max" ])
        #print(naive_segments)
        pca = PCA(n_components=1)
        new_segs = pca.fit_transform(naive_segments)
        new_segs = new_segs.flatten()
        #print(new_segs)
        print(i)
        X.append(new_segs)
        y[i] = label
    X = pd.DataFrame(X, columns  = [f"feature {i}" for i in range(k)])
    return (X,y)

path = "haircut_more_data"
rough_dataset = ImageFolder(root = path)

#Commented out code to extract the below table once
#print(len(rough_dataset))
# X,y = feature_extraction(rough_dataset)
# #print(len(rough_dataset))
# fn = 'features_of_images_v2.csv'
# filepath = Path(fn)
# filepath.parent.mkdir(parents=True, exist_ok=True)
# names = [f"Segment {i}" for i in range(50)]
# dataset = pd.DataFrame(np.column_stack((X,y)), columns = names.append("labels"))
# dataset.to_csv(filepath, index=False)

#Opens the file created by commented out code above
filename = 'features_of_images_v2.csv' #v2 calls my second attempt, "features_of_images.csv" calls the first attempt
with open(filename, mode ='r') as file:
        df = pd.read_csv(file)
#print(df)
names = [f"Segment {i}" for i in range(20)] #for whatever reason, dataframe names were not saved to the file
names.append("labels")
df.columns = names
df = df.sample(frac = 1) #shuffling dataset
#print(df)
X = df.drop("labels", axis = 1)
y = df["labels"]

#random train/test 70/30 split
X_train = X.iloc[1:int(.7*len(X))]
y_train = y[1:int(.7*len(y))]
X_test = X.iloc[int(.7*len(X))+1:len(X)]
y_test = y[int(.7*len(y))+1:len(y)]

s = make_pipeline(StandardScaler(), svm.SVC())
s.fit(X_train,y_train)

#train errors
preds = s.predict(X_train)
mspe = sum((preds - y_train)**2 )/len(y_train)
accuracy = sum((preds == y_train))/len(y_train)
print(mspe)
print(accuracy)
#Test errors
preds = s.predict(X_test)
mspe = sum((preds - y_test)**2 )/len(y_test)
accuracy = sum((preds == y_test))/len(y_test)
print(mspe)
print(accuracy)
