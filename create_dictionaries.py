#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 11:38:00 2023

@author: oliveira
"""

# %% Import libraries
import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

path = '/data/seismo/oliveira/Irradiance/Machine_Learning/Large_Dataset/'
folder = 'Dataset_Augmented/'


# %% Create dictionaries for all images

# Clear old files

partition_file = os.path.join(path, "partition.json")  
if os.path.isfile(partition_file):
    os.remove(partition_file)

labels_file = os.path.join(path, "labels.json")  
if os.path.isfile(labels_file):
    os.remove(labels_file)

# Write partition dictionary

partition = {}

mask_files = [os.path.split(f)[-1][:-4] # [:-4] removes the file extension
                    for f in glob.glob(path + folder + "mag_*")]

# random.shuffle(mask_files)

files_validation = mask_files[:1200]
files_train = list(set(mask_files) - set(files_validation))

partition["train"] = files_train
partition["validation"] = files_validation

with open(partition_file, "w") as fp:
    json.dump(partition, fp)

# Write labels dictionary

labels = {}

for file in mask_files:
    label = "mask_ic" + file[3:]
    labels[file] = label

with open(labels_file, "w") as fp:
    json.dump(labels, fp)


# %% Select a subset of images with large spots

all_files = [os.path.split(f)[-1][:-4]
                for f in glob.glob(path + folder + "*")]

mask_files = [i for i in all_files if i.startswith("mask_")]
mag_files = [i for i in all_files if i.startswith("mag_")]

#mask_files = sorted(mask_files)
#mag_files = sorted(mag_files)

n_samples = len(mask_files)

# Making sure that the mask and mag files are in the same order
for i in range(n_samples):
    a = mask_files[i][7:]
    b = mag_files[i][3:]
    if a != b:
        print(a, b)


# %% Delete old dictionaries

partition_file = os.path.join(path, "partition_subset.json")  
if os.path.isfile(partition_file):
    os.remove(partition_file)

labels_file = os.path.join(path, "labels_subset.json")  
if os.path.isfile(labels_file):
    os.remove(labels_file)


# %% Create dictionaries for the subset of images

labels = {}

for i in range(n_samples):
    labels[mag_files[i]] = mask_files[i]

with open(labels_file, "w") as fp:
    json.dump(labels, fp)


partition = {}
train = []
validation = []

index_validation = np.random.choice(n_samples, n_samples//10, replace=False)
index_training = [x for x in range(n_samples) if x not in index_validation]
np.random.shuffle(index_training)

for i in index_training:
    train.append(mag_files[i])

for i in index_validation:
    validation.append(mag_files[i])

partition["train"] = train
partition["validation"] = validation

with open(partition_file, "w") as fp:
    json.dump(partition, fp)


# %%
