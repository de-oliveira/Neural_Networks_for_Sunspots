#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 14:47:19 2023

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
with open('partition.json') as f:
    data = f.read()
partition = json.loads(data)

with open('labels.json') as f:
    data = f.read()
labels = json.loads(data)

# Generators
folder = 'Data_All_Slices/'

training_set = Dataset(partition['train'], labels, folder)
training_loader = torch.utils.data.DataLoader(training_set, **params)

validation_set = Dataset(partition['validation'], labels, folder)
validation_loader = torch.utils.data.DataLoader(validation_set, **params)

for i in range(20):
    plt.imshow(training_set[i][0])
    plt.colorbar()
    plt.show()
    print("Spots? ", training_set[i][1])


# %% Defining parameters, class and functions

IMAGE_WIDTH = 128
INPUT_SIZE = IMAGE_WIDTH * IMAGE_WIDTH
OUTPUT_SIZE = 2
BATCH_SIZE = 10
NUM_EPOCHS = 5
LEARNING_RATE = 3.0

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_layer = nn.Linear(INPUT_SIZE, 200)
        self.output_layer = nn.Linear(200, OUTPUT_SIZE)

    def forward(self, x):
        x = x.reshape(-1, INPUT_SIZE)
        x = torch.sigmoid(self.hidden_layer(x))
        x = torch.sigmoid(self.output_layer(x))
        return x

def expand_expected_output(tensor_of_expected_outputs, output_size):
    return torch.tensor([expand_single_output(expected_output.item(),
                                              output_size)
                         for expected_output in tensor_of_expected_outputs])


def expand_single_output(expected_output, output_size):
    x = [0.0 for _ in range(output_size)]
    x[expected_output] = 1.0
    return x


def create_loss_function(loss_function, output_size=OUTPUT_SIZE):
    def calc_loss(outputs, target):
        targets = expand_expected_output(target, output_size)
        return loss_function(outputs, targets)
    return calc_loss


def train_network(model, data_loader, test_loader, num_epochs, loss_function, optimizer):
    for epoch in range(num_epochs):
        model = model.train()
        for batch in enumerate(data_loader):
            i, (images, expected_outputs) = batch
            images = images.to(torch.float32)
            # print(images.dtype)
            # print(type(images))
            # print(images.shape)
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
            # print(images.dtype)
            # print(type(images))
            # print(images.shape)
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

net = Net()

mse_loss_function = nn.MSELoss()
loss_function = create_loss_function(mse_loss_function)
sgd = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE)

train_network(net, training_loader, validation_loader, NUM_EPOCHS, loss_function, sgd)
