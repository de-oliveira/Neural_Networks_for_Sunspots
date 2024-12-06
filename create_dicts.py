#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 15:18:40 2023

@author: oliveira
"""

import json
import glob
import os


# %% Clear old files

path = '/data/seismo/oliveira/Irradiance/Machine_Learning/Spot_Masks/'

partition_file = os.path.join(path, "partition_normalized.json")  
if os.path.isfile(partition_file):
    os.remove(partition_file)

labels_file = os.path.join(path, "labels_normalized.json")  
if os.path.isfile(labels_file):
    os.remove(labels_file)


# %% Write partition dictionary

folder = "./Data_Normalized/"

partition = {}

files_all = [os.path.split(f)[-1][:-4] # [:-4] removes the file extension
                    for f in glob.glob(folder + "*")]

files_validation = [os.path.split(f)[-1][:-4]
                    for f in glob.glob(folder + "mag_slice_*_year_2022*")]

files_train = list(set(files_all) - set(files_validation))

partition["train"] = files_train
partition["validation"] = files_validation

with open(partition_file, "w") as fp:
    json.dump(partition, fp)


# %% Write labels dictionary

folder_labels = "Labels_All_Data"

labels = {}
files_labels = glob.glob("./" + folder_labels + "/labels_*")

with open(files_labels[0]) as f:
    data = f.read()
    
js0 = json.loads(data)
labels = js0

for i in range(1, len(files_labels)):
    with open(files_labels[i]) as f:
        data = f.read()
    js1 = json.loads(data)
    labels = labels | js1

with open(labels_file, "w") as fp:
    json.dump(labels, fp)
