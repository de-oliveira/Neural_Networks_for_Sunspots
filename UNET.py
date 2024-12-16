'''
Created on Mon Oct 23 13:57:00 2023

@author: oliveira

UNET in Pytorch. Adapted from: 
https://medium.com/analytics-vidhya/unet-implementation-in-pytorch-idiot-developer-da40d955f201
https://github.com/ArdaGen/Multi_Class_Semantic_Segmentation

'''

# %% Import libraries
import torch
import torch.nn as nn
import gc
import json
from my_classes import Dataset
from my_NNs import UNet
import loss_functions as lf
import wandb
import matplotlib.pyplot as plt
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
s = 32
torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=device), 
                           torch.zeros(s, s, s, s, device=device))
#model = UNet().to(device)


# %% Data sets and data loaders

# # Parameters
# params = {'batch_size': 70,
#           'shuffle': True,
#           'num_workers': 6}

# # Dictionaries
# with open('partition.json') as f:
#     data = f.read()
# partition = json.loads(data)

# with open('labels.json') as f:
#     data = f.read()
# labels = json.loads(data)

# # Generators
# path = 'Datasets_1024x1024/'

# training_set = Dataset(partition['train'], labels, path)
# training_loader = torch.utils.data.DataLoader(training_set, **params)

# validation_set = Dataset(partition['validation'], labels, path)
# validation_loader = torch.utils.data.DataLoader(validation_set, **params)


# %% Data sets and data loaders for a subset of images

batch_size = 10
params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 6}

with open('partition_subset.json') as f:
    data = f.read()
partition = json.loads(data)

with open('labels_subset.json') as f:
    data = f.read()
labels = json.loads(data)

# Generators
path = 'Dataset_Augmented/'

training_set = Dataset(partition['train'], labels, path)
training_loader = torch.utils.data.DataLoader(training_set, **params)

validation_set = Dataset(partition['validation'], labels, path)
validation_loader = torch.utils.data.DataLoader(validation_set, **params)


# %% Plotting

for i in range(2):
    fig, ax = plt.subplots(2)
    plt.subplots_adjust(hspace = 0., wspace = 0.)
    
    ax1 = plt.subplot(211)
    im1 = plt.imshow(training_set[i][0], cmap = 'gray')
    
    ax2 = plt.subplot(212)
    im2 = plt.imshow((training_set[i][1]), cmap = 'gray')

    plt.colorbar(im1, ax=ax1)
    plt.colorbar(im2, ax=ax2)

    plt.show()


# %% Training and testing model

def train_network(model, training_loader, test_loader, num_epochs, loss_function, optimizer):
    for epoch in range(num_epochs):
        model = model.train()

        for i, (images, labels) in enumerate(training_loader):
            images = abs(images).to(torch.float32)
            images = images.reshape((-1, 1, 128, 128)).to(device)

            labels = labels.to(torch.float32)
            labels = labels.reshape((-1, 1, 128, 128)).to(device)

            # Forward pass
            torch.cuda.empty_cache()
            outputs = model(images).to(device)
            # Comment the line below if using function other than BCELoss
            outputs = nn.Sigmoid()(outputs)
            loss = loss_function(outputs, labels).to(device)

            # Expected loss
            loss_expected = loss_function(images, labels).to(device)
            blank = torch.zeros_like(images).to(device)
            loss_blank = loss_function(blank, labels).to(device)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            del images, labels, outputs
            torch.cuda.empty_cache()
            gc.collect()
        
        wandb.log({"train_loss": loss, "train_loss_expected": loss_expected, "train_loss_blank": loss_blank})
        epoch_info = 'Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item())

        test_image_list, test_expected, test_predicted = test_network(model, test_loader, 
                                                                 loss_function, epoch_info)

    return test_image_list, test_expected, test_predicted


def test_network(model, data_loader, loss_function, epoch_info):
    model = model.eval()
    with torch.no_grad():
        test_expected = []
        test_predicted = []
        test_image_list = []

        for images, labels in data_loader:
            images = abs(images).to(torch.float32)
            images = images.reshape((-1, 1, 128, 128)).to(device)

            labels = labels.to(torch.float32)
            labels = labels.reshape((-1, 1, 128, 128)).to(device)

            outputs = model(images).to(device)
            # Comment the line below if using function other than BCELoss
            outputs = nn.Sigmoid()(outputs)
            loss = loss_function(outputs, labels).to(device)

            test_expected.append(labels)
            test_predicted.append(outputs)
            test_image_list.append(images)

            loss_expected = loss_function(images, labels).to(device)
            blank = torch.zeros_like(images).to(device)
            loss_blank = loss_function(blank, labels).to(device)

            del images, labels, outputs
            torch.cuda.empty_cache()
            gc.collect()

        wandb.log({"test_loss": loss, "test_loss_expected": loss_expected, "test_loss_blank": loss_blank})
        print(epoch_info)
    return test_image_list, test_expected, test_predicted 


# %% Defining hyperparamters

# Define model
num_epochs = 10
model = UNet().to(device)

# Loss and optimizer
#criterion = lf.FocalLoss(alpha=0.2, gamma=1).to(device)
#criterion = lf.DiceBCELoss().to(device)
criterion = nn.BCELoss().to(device)

learning_rate = 0.0005
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Initiate wandb
wandb.init(
    project="where-are-the-spots",
    config={
    "architecture": "UNet",
    "epochs": num_epochs,
    "loss_function": "BCELoss",
    "learning_rate": learning_rate,
    })

# Run training and testing
test_images, test_exp, test_pred = train_network(model, training_loader, 
                        validation_loader, num_epochs, criterion, optimizer)

wandb.finish()


 # %% Plotting results for test

test_images0 = test_images[-2].reshape(-1,128,128).cpu().numpy()
test_exp0 = test_exp[-2].reshape(-1,128,128).cpu().numpy()
test_pred0 = test_pred[-2].reshape(-1,128,128).cpu().numpy()

for i in range(3):
    fig, ax = plt.subplots(3, figsize=(8,10))
    im0 = ax[0].imshow(test_images0[i], cmap='grey', vmin=0, vmax=1)
    ax[0].set_title('$|B_{LOS}|$')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.colorbar(im0)

    im1 = ax[1].imshow(test_exp0[i], cmap='grey', vmin=0, vmax=1)
    ax[1].set_title('Ground Truth Spot Mask')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    plt.colorbar(im1)

    im2 = ax[2].imshow(test_pred0[i], cmap='grey', vmin=0, vmax=1)
    ax[2].set_title('Predicted Spot Mask')
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    plt.colorbar(im2)

    plt.tight_layout()
    plt.show()


# Plot flattened images
for i in range(3):
    fig, ax = plt.subplots(figsize=(6,8))
    plt.plot(test_images0[i].flatten(), label='Image')
    plt.plot(test_exp0[i].flatten(), label='Ground Truth')
    plt.plot(test_pred0[i].flatten(), label='Prediction')
    plt.legend()
    plt.show()

# Setting a treshold for the predicted images
threshold = 0.5
spots = test_pred0 >= threshold
background = test_pred0 < threshold
predicted = np.copy(test_pred0)
predicted[spots] = 1
predicted[background] = 0

for i in range(3):
    fig, ax = plt.subplots(2, 2, figsize=(10,10))
    im0 = ax[0,0].imshow(test_images0[i], cmap='grey', vmin=0, vmax=1)
    ax[0,0].set_title('$|B_{LOS}|$')
    ax[0,0].set_xticks([])
    ax[0,0].set_yticks([])

    im1 = ax[0,1].imshow(test_exp0[i], cmap='grey', vmin=0, vmax=1)
    ax[0,1].set_title('Ground Truth Spot Mask')
    ax[0,1].set_xticks([])
    ax[0,1].set_yticks([])

    im2 = ax[1,0].imshow(test_pred0[i], cmap='grey', vmin=0, vmax=1)
    ax[1,0].set_title('Predicted Spot Mask')
    ax[1,0].set_xticks([])
    ax[1,0].set_yticks([])

    im3 = ax[1,1].imshow(predicted[i], cmap='grey', vmin=0, vmax=1)
    ax[1,1].set_title('Predicted Spot Mask (treshold = {})'.format(threshold))
    ax[1,1].set_xticks([])
    ax[1,1].set_yticks([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    cax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
    cbar = plt.colorbar(im0, cax=cax, orientation='horizontal')
    cbar.set_label('Area Coverage')
    plt.show()

for i in range(10):
    fig, ax = plt.subplots(3, figsize=(8,10))
    im0 = ax[0].imshow(test_images0[i], cmap='grey', vmin=0, vmax=1)
    ax[0].set_title('$|B_{LOS}|$')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.colorbar(im0)

    im1 = ax[1].imshow(test_exp0[i], cmap='grey', vmin=0, vmax=1)
    ax[1].set_title('Ground Truth Spot Mask')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    plt.colorbar(im1)

    im2 = ax[2].imshow(predicted[i], cmap='grey', vmin=0, vmax=1)
    ax[2].set_title('Predicted Spot Mask')
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    plt.colorbar(im2)

    plt.tight_layout()
    plt.show()


# Testing if results are all zeros
#for i in range(len(test_pred)-1):
#    a = test_pred[i].cpu().numpy().reshape(batch_size, 128, 128)
#    print(np.sum(a))


# %% Testing model on a single image

model = UNet()
criterion = lf.DiceBCELoss()
learning_rate = 0.0005
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

image = abs(training_set[0][0]).reshape((-1, 1, 128, 128)).to(torch.float32)
label = training_set[0][1].reshape((-1, 1, 128, 128)).to(torch.float32)

outputs = model(image)
loss = criterion(outputs, labels)

# The line below gives me the INDEXES of the maximum values in the tensor,
# which is NOT what I want. I want the values themselves.
# _, predicted_outputs = torch.max(outputs.data, 1)

fig, ax = plt.subplots(3)
ax[0].imshow(image.reshape(128,128), cmap='grey')
ax[1].imshow(label.reshape(128,128), cmap='grey')
ax[2].imshow(outputs.detach().numpy().reshape(128,128), cmap='grey')


# %%
