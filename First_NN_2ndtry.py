#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 16:15:00 2023

@author: oliveira
"""

# %% Import libraries

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import torch.optim as optim
from my_classes import Dataset
import matplotlib.pyplot as plt
import json
import wandb
import numpy as np


# %% Data sets and data loaders

# CUDA for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
params = {'batch_size': 10,
          'shuffle': True,
          'num_workers': 6}

# Dictionaries
with open('partition.json') as f:
    data = f.read()
partition = json.loads(data)

with open('labels.json') as f:
    data = f.read()
labels = json.loads(data)

# Generators
path = 'Datasets_1024x1024/'

training_set = Dataset(partition['train'], labels, path)
training_loader = torch.utils.data.DataLoader(training_set, **params)

validation_set = Dataset(partition['validation'], labels, path)
validation_loader = torch.utils.data.DataLoader(validation_set, **params)

for i in range(20):
    fig, ax = plt.subplots(2, figsize = (8, 11))
    plt.subplots_adjust(hspace = 0., wspace = 0.)
    
    ax1 = plt.subplot(211)
    im1 = plt.imshow(training_set[i][0], cmap = 'gray')
    
    ax2 = plt.subplot(212)
    im2 = plt.imshow(training_set[i][1], cmap = 'gist_heat')

    plt.show()


# %% Defining classes and functions

class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 'same'),
                        nn.BatchNorm2d(256),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 'same'),
                        nn.BatchNorm2d(256))
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block):
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.BatchNorm2d(1),
                        nn.Conv2d(1, 256, kernel_size = 3, stride = 1, padding = 'same'),
                        nn.BatchNorm2d(256),
                        nn.ReLU())
        self.residual_blocks = self._make_layer(ResidualBlock, 10)
        self.conv2 = nn.Conv2d(256, 1, kernel_size = 3, stride = 1, padding = 'same')
        
    def _make_layer(self, block, blocks):
        layers = []
        for i in range(blocks):
            layers.append(block())
        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.residual_blocks(x)
        x = self.conv2(x)

        return x


# %% Defining training and testing of the model

def train_network(model, data_loader, test_loader, num_epochs, loss_function, optimizer):

    for epoch in range(num_epochs):
        model = model.train()
        correct = 0
        total = 0

        for batch in data_loader:
            images, expected_outputs = batch
            images = images.to(torch.float32)
            images = images.reshape((-1, 1, 128, 128))
            expected_outputs = expected_outputs.to(torch.float32)
            expected_outputs = expected_outputs.reshape((-1, 1, 128, 128))

            outputs = model(images)

            predicted_outputs = torch.argmax(outputs, dim=1)
            correct += (predicted_outputs == expected_outputs).sum()
            total += expected_outputs.size(0)

            loss = loss_function(outputs, expected_outputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        accuracy_train = float(correct)/total
        wandb.log({"accuracy_train": accuracy_train, "loss_train": loss})

        epoch_info = "Epoch {}/{}".format(epoch+1, num_epochs)
        image_list, test_expected, test_predicted = test_network(model, test_loader, epoch_info)
    return image_list, test_expected, test_predicted


def test_network(model, data_loader, epoch_info=""):
    model = model.eval()
    with torch.no_grad():
        correct = 0
        total = 0

        expected = []
        predicted = []
        image_list = []

        for batch in data_loader:
            images, expected_outputs = batch
            images = images.to(torch.float32)
            images = images.reshape((-1, 1, 128, 128))
            expected_outputs = expected_outputs.to(torch.float32)
            expected_outputs = expected_outputs.reshape((-1, 1, 128, 128))
            
            outputs = model(images)

            # get the predicted value from each output in the batch
            predicted_outputs = torch.argmax(outputs, dim=1)

            correct += (predicted_outputs == expected_outputs).sum()
            total += expected_outputs.size(0)
            
            expected.append(expected_outputs)
            predicted.append(predicted_outputs)
            image_list.append(images)
                    
        accuracy_test = float(correct)/total
        wandb.log({"accuracy_test": accuracy_test})

        results_str = f"Test data results: {accuracy_test}"
        if epoch_info:
            results_str += f", {epoch_info}"
        print(results_str)
    return image_list, expected, predicted 


# %% Setting hyperparameters

num_epochs = 30
batch_size = 10
learning_rate = 0.001
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-08

net = ResNet(ResidualBlock)

# Loss and optimizer
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate, betas = (beta_1, beta_2), eps = epsilon)  

# Start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="where-are-the-spots",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": learning_rate,
    "architecture": "ResNet",
    "epochs": num_epochs,
    }
)

img, exp, pred = train_network(net, training_loader, validation_loader, num_epochs, loss_function, optimizer)

wandb.finish()
# %%
