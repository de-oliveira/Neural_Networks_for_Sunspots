#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 16:32:22 2023

@author: oliveira
"""

import torch
import numpy as np

file_extension = '.npy'

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels, folder):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.path = folder

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = npy_loader(self.path + ID + file_extension)
        y = self.labels[ID]

        return X, y
    
def npy_loader(path):
    sample = torch.from_numpy(np.load(path))
    return sample