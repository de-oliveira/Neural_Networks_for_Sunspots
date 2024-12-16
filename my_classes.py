#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Oct 16 16:16:00 2023

@author: oliveira
"""

import torch
import numpy as np


class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  
  def __init__(self, list_IDs, labels, folder, file_extension='.npy', transform=None):
    """
    Parameters
    ----------
    list_IDs : dict
        Dictionary with keys 'training' and 'validation', and respective values.
    labels : dict
        Dictionary with the keys being the IDs of mag and values being the IDs of continuum.
    folder : str
        The path to the dictionaries.
    file_extension : str, optional
        File extension of the data, by default '.npy'
    """

    'Initialization'
    self.labels = labels
    self.list_IDs = list_IDs
    self.path = folder
    self.extension = file_extension
    self.transform = transform


  def __len__(self):
    'Denotes the total number of samples'
    return len(self.list_IDs)

  def __getitem__(self, index):
    'Generates one sample of data'
    # Select sample
    ID = self.list_IDs[index]

    # Load data and get label
    X = npy_loader(self.path + ID + self.extension)
    Y = npy_loader(self.path + self.labels[ID] + self.extension)

    # Apply transform
    if self.transform:
        X = self.transform(X)
        Y = self.transform(Y)
    return X, Y
  
  

def npy_loader(path):
    sample = torch.from_numpy(np.load(path))
    return sample
