'''ResNet in PyTorch.

Adapted from:
https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/
'''

# %% Import libraries

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import gc
import json
from my_classes import Dataset


# %% Data sets and data loaders

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
params = {'batch_size': 70,
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
path = 'Datasets_1024x1024/'

training_set = Dataset(partition['train'], labels, path)
training_loader = torch.utils.data.DataLoader(training_set, **params)

validation_set = Dataset(partition['validation'], labels, path)
validation_loader = torch.utils.data.DataLoader(validation_set, **params)

# for i in range(20):
#     fig, ax = plt.subplots(2, figsize = (8, 11))
#     plt.subplots_adjust(hspace = 0., wspace = 0.)
    
#     ax1 = plt.subplot(211)
#     im1 = plt.imshow(training_set[i][0], cmap = 'gray')
    
#     ax2 = plt.subplot(212)
#     im2 = plt.imshow(training_set[i][1], cmap = 'gist_heat')

#     plt.show()


# %% Defining classes and functions

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
    def __init__(self, block, layers):
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
        self.fc = nn.Linear(512, 1)
        
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


# %% Setting hyperparameters

num_epochs = 20

model = ResNet(ResidualBlock, [2, 2, 2, 2]).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, weight_decay = 0.001, momentum = 0.9)  


# %% Training and validation

list_of_predicted = []
num_epochs = 2

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(training_loader):  
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)
        
        images = images.to(torch.float32)
        images = images.reshape((-1, 1, 128, 128))

        labels = labels.to(torch.float32)
        labels = labels.reshape((-1, 1, 128, 128))
        
        # print(images.shape)
        # print(labels.shape)

        # Forward pass
        outputs = model(images)

        # print(outputs.shape)
        loss = criterion(outputs, labels.view(1, -1))
        
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

            labels = labels.to(torch.float32)
            labels = labels.reshape((-1, 1, 128, 128))

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            print(predicted.shape)
            print(outputs.shape)
            print(labels.shape)

            correct += (predicted == labels[1]).sum().item()
            list_of_predicted.append(outputs)
            del images, labels, outputs
    
        print('Accuracy: {} %'.format(100 * correct / total))


# %%
