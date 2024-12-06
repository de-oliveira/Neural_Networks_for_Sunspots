#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 16:07:00 2023

@author: oliveira
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from my_classes import Dataset
import matplotlib.pyplot as plt
import os
import numpy as np
import json


# %% Data sets and data loaders

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Parameters
params = {'batch_size': 10,
          'shuffle': True,
          'num_workers': 6}
max_epochs = 100

# Dictionaries
with open('partition_normalized.json') as f:
    data = f.read()
partition = json.loads(data)

with open('labels_normalized.json') as f:
    data = f.read()
labels = json.loads(data)

# Generators
folder = '/data/seismo/oliveira/Irradiance/Machine_Learning/Spot_Masks/Data_Normalized/'

training_set = Dataset(partition['train'], labels, folder)
training_loader = torch.utils.data.DataLoader(training_set, **params)

validation_set = Dataset(partition['validation'], labels, folder)
validation_loader = torch.utils.data.DataLoader(validation_set, **params)

len_train = len(training_set) # 47480 images
len_test = len(validation_set) # 4333 images ->> 8% of all data are testing data

len_data = len(labels.values())
len_0 = np.unique(list(labels.values()), return_counts=True)[1][0]
len_1 = np.unique(list(labels.values()), return_counts=True)[1][1]
ratio_0 = len_0/len_data # 87% of images without spots
ratio_1 = len_1/len_data # 13% of images with spots

for i in range(20):
    plt.imshow(training_set[i][0])
    plt.colorbar()
    plt.show()
    print("Spots? ", training_set[i][1])


# %% Defining class and functions

class ConvNetSimple(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5)
        self.fc1 = nn.Linear(62*62*20, 100)
        self.out = nn.Linear(100, OUTPUT_SIZE)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.sigmoid(x)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        x = x.view(-1, 62*62*20)
        x = self.fc1(x)
        x = torch.sigmoid(x)

        x = self.out(x)
        return x


def train_network(model, data_loader, test_loader, num_epochs, loss_function, optimizer):
    for epoch in range(num_epochs):
        model = model.train()
        for batch in enumerate(data_loader):
            i, (images, expected_outputs) = batch
            images = images.to(torch.float32)
            images = images.reshape((-1, 1, 128, 128))

            outputs = model(images)
            loss = loss_function(outputs, expected_outputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_info = "Epoch {}/{}".format(epoch+1, num_epochs)
        test_network(model, test_loader, epoch_info)


def test_network(model, data_loader, epoch_info=""):
    model = model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in data_loader:
            images, expected_outputs = batch
            images = images.to(torch.float32)

            images = images.reshape((-1, 1, 128, 128))

            outputs = model(images)

            # get the predicted value from each output in the batch
            predicted_outputs = torch.argmax(outputs, dim=1)
            correct += (predicted_outputs == expected_outputs).sum()
            total += expected_outputs.size(0)

        results_str = f"Test data results: {float(correct)/total}"
        if epoch_info:
            results_str += f", {epoch_info}"
        print(results_str)


# %% Running script

IMAGE_WIDTH = 128
INPUT_SIZE = IMAGE_WIDTH * IMAGE_WIDTH
OUTPUT_SIZE = 2
BATCH_SIZE = 10
NUM_EPOCHS = 10
LEARNING_RATE = 0.1
WEIGTH_DECAY = 0

net = ConvNetSimple()

loss_function=nn.CrossEntropyLoss()
sgd = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGTH_DECAY)

train_network(net, training_loader, validation_loader, NUM_EPOCHS, loss_function, sgd)


# %%
