#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 16:07:00 2023

@author: oliveira
"""
# %% Import libraries

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from my_classes import Dataset
import matplotlib.pyplot as plt
import os
import numpy as np
import json
import random
import wandb
import gc


# %% Data sets and data loaders

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Parameters
params = {'batch_size': 70,
          'shuffle': True,
          'num_workers': 6}
max_epochs = 100

# Dictionaries
# with open('partition_normalized.json') as f:
#     data = f.read()
# partition = json.loads(data)

with open('labels_normalized.json') as f:
    data = f.read()
labels = json.loads(data)

# Homogenize classes
values = list(labels.values())

indexes_0 = [i for i, n in enumerate(values) if n == 0]
indexes_1 = [i for i, n in enumerate(values) if n == 1]

list_without_spots = []

for index in indexes_0:
    list_without_spots.append(list(labels)[index])

list_with_spots = []

for index in indexes_1:
    list_with_spots.append(list(labels)[index])

new_list_without_spots = list_without_spots[0:-1:7]
random.shuffle(new_list_without_spots)

full_list = list_with_spots + new_list_without_spots # 6911 + 6415 data points
random.shuffle(full_list)   # Randomize list

partition = {}
partition["train"] = full_list[:10000]          # 75% of the data points
partition["validation"] = full_list[10000:]     # 25% of the data points

# Generators
folder = 'Data_Normalized/'

training_set = Dataset(partition['train'], labels, folder)
training_loader = torch.utils.data.DataLoader(training_set, **params)

validation_set = Dataset(partition['validation'], labels, folder)
validation_loader = torch.utils.data.DataLoader(validation_set, **params)

# for i in range(20):
#     plt.imshow(training_set[i][0])
#     plt.colorbar()
#     plt.show()
#     print("Spots? ", training_set[i][1])


# %% Defining class and functions

class ConvNetTwoConvLayersReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=5)
        self.fc1 = nn.Linear(29*29*40, 100)
        self.out = nn.Linear(100, OUTPUT_SIZE)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        x = x.view(-1, 29*29*40)
        x = self.fc1(x)
        x = torch.relu(x)

        x = self.out(x)
        return x


class ConvNetThreeConvLayersReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=40, out_channels=80, kernel_size=6)
        self.fc1 = nn.Linear(12*12*80, 100)
        self.out = nn.Linear(100, OUTPUT_SIZE)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv3(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        x = x.view(-1, 12*12*80)
        x = self.fc1(x)
        x = torch.relu(x)

        x = self.out(x)
        return x


class ConvNetFourConvLayersReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=40, out_channels=60, kernel_size=6)
        self.conv4 = nn.Conv2d(in_channels=60, out_channels=80, kernel_size=5)
        self.fc1 = nn.Linear(4*4*80, 100)
        self.out = nn.Linear(100, OUTPUT_SIZE)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv3(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv4(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        x = x.view(-1, 4*4*80)
        x = self.fc1(x)
        x = torch.relu(x)

        x = self.out(x)
        return x
    

class ConvNetFiveConvLayersReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=40, out_channels=60, kernel_size=6)
        self.conv4 = nn.Conv2d(in_channels=60, out_channels=80, kernel_size=5)
        self.conv5 = nn.Conv2d(in_channels=80, out_channels=100, kernel_size=3)
        self.fc1 = nn.Linear(1*1*100, 100)
        self.out = nn.Linear(100, OUTPUT_SIZE)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv3(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv4(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv5(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        x = x.view(-1, 1*1*100)
        x = self.fc1(x)
        x = torch.relu(x)

        x = self.out(x)
        return x


class ConvNetSixConvLayersReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2**4, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=2**4, out_channels=2**5, kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=2**5, out_channels=2**6, kernel_size=2)
        self.conv4 = nn.Conv2d(in_channels=2**6, out_channels=2**7, kernel_size=2)
        self.conv5 = nn.Conv2d(in_channels=2**7, out_channels=2**8, kernel_size=2)
        self.conv6 = nn.Conv2d(in_channels=2**8, out_channels=2**9, kernel_size=2)
        self.fc1 = nn.Linear(1*1*512, 100)
        self.out = nn.Linear(100, OUTPUT_SIZE)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv3(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv4(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv5(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv6(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        x = x.view(-1, 1*1*512)
        x = self.fc1(x)
        x = torch.relu(x)

        x = self.out(x)
        return x
    

class ConvNetTwoConvTwoDenseLayersWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=5)

        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(29*29*40, 1000)

        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1000, 1000)

        self.dropout3 = nn.Dropout(p=0.5)
        self.out = nn.Linear(1000, OUTPUT_SIZE)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, kernel_size=2, stride=2)

        x = x.view(-1, 29*29*40)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = torch.relu(x)

        x = self.dropout2(x)
        x = self.fc2(x)
        x = torch.relu(x)

        x = self.dropout3(x)
        x = self.out(x)
        return x
    

class CNNClassifier(nn.Module):
    def __init__(self):
        ks = 2
        super(CNNClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=ks, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=ks, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        
            nn.Conv2d(32, 64, kernel_size=ks, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64 ,64, kernel_size=ks, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=ks, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=ks, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Flatten(),
            nn.Linear(3200, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, xb):
        return self.network(xb)


# %% Defining training and testing

def train_network(model, data_loader, test_loader, num_epochs, loss_function, optimizer):

    for epoch in range(num_epochs):
        model = model.train()
        correct = 0
        total = 0

        for batch in data_loader:
            images, expected_outputs = batch
            images = images.to(torch.float32)
            images = images.reshape((-1, 1, 128, 128))

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
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        expected = []
        predicted = []
        image_list = []

        for batch in data_loader:
            images, expected_outputs = batch
            images = images.to(torch.float32)
            images = images.reshape((-1, 1, 128, 128))

            outputs = model(images)

            # get the predicted value from each output in the batch
            predicted_outputs = torch.argmax(outputs, dim=1)

            correct += (predicted_outputs == expected_outputs).sum()
            total += expected_outputs.size(0)
            
            expected.append(predicted_outputs)
            predicted.append(predicted_outputs)
            image_list.append(images)

            # get confusion matrix
            for i in range(len(expected_outputs)):
                if predicted_outputs[i] == expected_outputs[i]:
                    if predicted_outputs[i] == 1:
                        TP += 1
                    else:
                        TN += 1
                else:
                    if predicted_outputs[i] == 1:
                        FP += 1
                    else:
                        FN += 1
                    
        accuracy_test = float(correct)/total
        wandb.log({"accuracy_test": accuracy_test, "true_positive": TP, 
                   "true_negative": TN, "false_positive": FP, "false_negative": FN})

        results_str = f"Test data results: {accuracy_test}"
        if epoch_info:
            results_str += f", {epoch_info}"
        print(results_str)
    return image_list, expected, predicted


# %% Running ResNet

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

  
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(1, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(4, stride = 2)
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

num_classes = 2
model = ResNet(ResidualBlock, [2, 2, 2, 2], num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay = 0.001, momentum = 0.9)  


# %% Training the ResNet

num_epochs = 2

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(training_loader):  
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)
        
        images = images.to(torch.float32)
        images = images.reshape((-1, 1, 128, 128))

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del images, labels, outputs
        torch.cuda.empty_cache()
        gc.collect()

    print ('Epoch [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, loss.item()))
            
    # Validation
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in validation_loader:
            images = images.to(device)
            labels = labels.to(device)

            images = images.to(torch.float32)
            images = images.reshape((-1, 1, 128, 128))

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs
    
        print('Accuracy of the network on the validation images: {} %'.format(100 * correct / total)) 


# %% Running script

IMAGE_WIDTH = 128
INPUT_SIZE = IMAGE_WIDTH * IMAGE_WIDTH
OUTPUT_SIZE = 2
BATCH_SIZE = 10
NUM_EPOCHS = 30
LEARNING_RATE = 0.07
WEIGTH_DECAY = 0.005

net = ConvNetSixConvLayersReLU()

loss_function = nn.CrossEntropyLoss()
sgd = torch.optim.SGD(net.parameters(), lr = LEARNING_RATE, weight_decay = WEIGTH_DECAY)

# Start a new wandb run to track this script

wandb.init(
    # set the wandb project where this run will be logged
    project="is-there-a-spot",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": LEARNING_RATE,
    "architecture": "CNN-6Layers",
    "notes": "",
    "epochs": NUM_EPOCHS,
    "weight_decay": WEIGTH_DECAY,
    }
)

img, exp, pred = train_network(net, training_loader, validation_loader, NUM_EPOCHS, loss_function, sgd)

wandb.finish()


# %% Plotting

f = 14
# batch = 0
# item = 0

for batch in range(len(img)):
    for item in range(len(img[batch])):

        if exp[batch][item] == pred[batch][item]:
            if exp[batch][item] == 1:
                identifier = 'TP'
            else:
                identifier = 'TN'
        else:
            if exp[batch][item] == 1:
                identifier = 'FP'
            else:
                identifier = 'FN'


        fig, ax = plt.subplots(figsize = (8, 6))
        im = plt.imshow(img[batch][item][0], cmap = 'gray', vmin = -0.5, vmax = 0.5)

        plt.text(0.2, 1.02, 'Spot? {}, Predicted? {}'.format(exp[batch][item], pred[batch][item]), 
                transform = ax.transAxes, fontsize = f)
        plt.xticks(fontsize = f)
        plt.yticks(fontsize = f)

        plt.subplots_adjust(right = 0.9)
        cbar_ax = fig.add_axes([0.81, 0.13, 0.02, 0.75])
        cbar = fig.colorbar(im, cax = cbar_ax, extend = 'both')
        cbar.set_label('LOS Magnetic Field (G)', fontsize = f)
        cbar.ax.tick_params(labelsize = f)

        # Find out image ID
        # Save figure with image ID + TN/TP/FN/TP

        plt.savefig("Figures_After_NN/{}_batch_{}_item_{}.png".format(identifier, batch, item), 
                    dpi = 200, bbox_inches = 'tight')


# %% Testing some things

# dir(validation_loader)

# for i in range(len(validation_loader)):
#     print(i)    # 333 batches

# batch = next(iter(validation_loader))
# images, expected_outputs = batch
# images = images.to(torch.float32)
# images = images.reshape((-1, 1, 128, 128))

# outputs = net(images)

# predicted_outputs = torch.argmax(outputs, dim=1)


