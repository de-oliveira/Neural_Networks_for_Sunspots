#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 11:28:26 2023

@author: oliveira
"""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from datetime import datetime
import json
import os

f = 14
start = datetime.now()
folder = 'Data_Normalized'
labels_folder = 'Labels_All_Data'


# %% Removing files
print("Deleting old files...")

path = '/data/seismo/oliveira/Irradiance/Machine_Learning/Spot_Masks/'
for filename in os.listdir(os.path.join(path, folder)): 
    file = os.path.join(path, folder, filename)  
    if os.path.isfile(file):
        os.remove(file)

for filename in os.listdir(os.path.join(path, labels_folder)): 
    file = os.path.join(path, labels_folder, filename)  
    if os.path.isfile(file):
        os.remove(file)


# %% Load dataset

years = [2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011]
# years = [2014]

for year in years:
    print(f"YEAR {year}")
    
    print("Loading dataset...")

    every_x_days = 2
    data_mag = np.load(path + 'Degraded_Datasets/magnetograms_1024x1024_year_{}_every_{}_days.npy'.format(year, 
                                                            every_x_days))
    data_ic = np.load(path + 'Degraded_Datasets/continuum_1024x1024_year_{}_every_{}_days.npy'.format(year, 
                                                            every_x_days))
    
    n_data = data_mag.shape[0]
    resolution = data_mag.shape[1]
    
    
    # %% Normalizing the data sets
    
    print("Normalizing data values...")
    data_same_range = np.empty_like(data_ic)
    
    for datapoint in range(n_data):
        data_same_range[datapoint, :, :] = data_ic[datapoint] - np.nanmean(data_ic[datapoint])
    
    normalization_factor = 2500. # in G
    data_normalized = data_mag / normalization_factor
    data_normalized[np.where(data_normalized > 1.)] = 1.
    data_normalized[np.where(data_normalized < -1.)] = -1.

    del data_mag


    # %% Create a mask with spots (1) and not a spot (0)
    
    print("Creating spot masks...")
    mask = np.zeros((n_data, resolution, resolution), int)
    
    for datapoint in range(n_data):
        zeros = np.zeros_like(data_same_range[datapoint], int)
        ones = np.ones_like(data_same_range[datapoint], int)
        
        spot = np.where(np.logical_and(data_same_range[datapoint] > -50000, 
                        data_same_range[datapoint] <= -15000), ones, zeros)
        mask[datapoint, :, :] = spot.astype(int)
    
    
    # %% Look at plots with mask on
    
    # print("Plotting:")
    # for datapoint in tqdm(range(0, 50, 5)):
        
    #     fig, ax = plt.subplots(2, figsize = (8, 11))
    #     plt.subplots_adjust(hspace = 0., wspace = 0.)
        
    #     ax1 = plt.subplot(211)
    #     im1 = plt.imshow(data_same_range[datapoint], cmap = 'gist_heat', vmin = -50000, vmax = 15000)
    #     plt.title('datapoint {}'.format(datapoint), fontsize = f)
    #     plt.xticks(visible = False)
    #     ax1.text(50, 100, 'Continuum Intensity', bbox = {'facecolor': 'white', 'pad': 10}, fontsize = f)
        
    #     ax2 = plt.subplot(212)
    #     plt.imshow(data_mag[datapoint], cmap = 'gray', vmin = -200, vmax = 200)
        
    #     a = np.where(mask[datapoint] == 1)
    #     ax2.scatter(a[1], a[0], c = 'r', s = 0.1, label = "Spot")
    #     ax2.text(50, 100, 'Magnetogram', bbox = {'facecolor': 'white', 'pad': 10}, fontsize = f)
    #     plt.legend(fontsize = f, markerscale = 20)
        
    #     plt.savefig(path + 'Figures/Mask_year_{}_datapoint_{}'.format(year, datapoint), dpi = 200, bbox_inches = 'tight')
    
    del data_ic
    
    
    # %% Divide magnetograms in 64 sub-images
    
    print("Slicing images...")
    n_slices = 64
    n_pixels_per_row = int(np.sqrt(resolution**2 / n_slices))
    
    data_mag_sliced = []
    mask_sliced = []
    
    for datapoint in range(n_data):
        for x in range(0, resolution, n_pixels_per_row):
            for y in range(0, resolution, n_pixels_per_row):
                tile = data_normalized[datapoint, x:x+n_pixels_per_row, y:y+n_pixels_per_row]
                data_mag_sliced.append(tile)
                
                tile = mask[datapoint, x:x+n_pixels_per_row, y:y+n_pixels_per_row]
                mask_sliced.append(tile)
    
    data_mag_sliced = np.reshape(data_mag_sliced, (n_data, n_slices, n_pixels_per_row, n_pixels_per_row))
    mask_sliced = np.reshape(mask_sliced, (n_data, n_slices, n_pixels_per_row, n_pixels_per_row))
    
    del data_normalized, mask
    
    
    # %% Cut the first two and last two rows of slices (desregard high latitudes)
    
    data_mag_sliced = data_mag_sliced[:, 16:48]
    mask_sliced = mask_sliced[:, 16:48]
    
    
    # %% Cutting the edges out (so we don't have data outside the solar disc)

    n_slices = data_mag_sliced.shape[1]
    boolean_array = np.empty((n_data, n_slices), int)

    for datapoint in range(n_data):
        for slices in range(n_slices):
            boolean_array[datapoint, slices] = ~np.isnan(data_mag_sliced[datapoint, slices]).any()

    data_final = data_mag_sliced[np.where(boolean_array)]
    mask_final = mask_sliced[np.where(boolean_array)]

    datapoints_array = np.array(range(n_data))[np.where(boolean_array)[0]]
    slices_array = np.array(range(n_slices))[np.where(boolean_array)[1]]
    new_len = len(datapoints_array)
    
    del data_mag_sliced, mask_sliced
    
    
    # %% Create boolean spot mask
    
    print("Creating boolean masks...")
    mask_boolean = np.zeros(new_len, int)

    for i in range(new_len):
        if np.sum(mask_final[i] > 0):
             mask_boolean[i] = 1
        else:
             mask_boolean[i] = 0

    del mask_final
    
    
    # %% Save dataset
    
    print("Saving datasets...")
    
    labels = {}
    
    for i in range(new_len):
        datapoint = datapoints_array[i]
        slices = slices_array[i]
        
        np.save(os.path.join(path, folder, 'mag_slice_{}_year_{}_number_{}'.format(slices, year, datapoint)), 
                data_final[i])
        
        labels["mag_slice_{}_year_{}_number_{}".format(slices, year, datapoint)] = int(mask_boolean[i])

    with open(os.path.join(path, labels_folder, f"labels_{year}.txt"), "w") as fp:
        json.dump(labels, fp)


# %% Finish script

print("Finished!")

end = datetime.now()
total_time = end - start
print("Script run time: " + str(total_time))