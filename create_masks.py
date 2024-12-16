"""
Created on Fri Oct 27 11:50:26 2023

@author: oliveira
"""

import matplotlib.pyplot as plt
import numpy as np
import os

folder = "./Datasets_1024x1024/"

# %% Create masks for each IC file

for filename in os.listdir(folder):
    if filename.startswith("ic_slice"):
        data_ic = np.load(folder + filename)

        mask = np.zeros_like(data_ic, int)
            
        zeros = np.zeros_like(data_ic, int)
        ones = np.ones_like(data_ic, int)

        spot = np.where(np.logical_and(data_ic > -50000, 
                        data_ic <= -6000), ones, zeros)
        mask[:, :] = spot.astype(int)

        np.save(folder + "mask_" + filename, mask)

