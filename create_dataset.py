#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 15:21:46 2023

@author: oliveira
"""

import os
from astropy.io import fits
import numpy as np
import IsabelaFunctions as isa
from tqdm import tqdm
from datetime import datetime

f = 14

days = 365
years = [2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011]

start = datetime.now()


# %% Initialize script

for year in years:
    print(f"YEAR {year}")

    ds_ic = 'hmi.Ic_45s[{}.01.01_TAI/{}d@1d]'.format(year, days)
    ds_mag = 'hmi.m_45s[{}.01.01_TAI/{}d@1d]'.format(year, days)

    print("Checking if data exist... ")

    KeyList = [
        ('DATE-OBS', float),    # Necessary for script
        ('QUALITY', str),       # Necessary for removing bad quality data
        ('CRPIX1', float),      # Necessary for cutting edges + limb darkening correction
        ('CRPIX2', float),      # Necessary for cutting edges + limb darkening correction
        ('CDELT1', float),      # Necessary for cutting edges + limb darkening correction
        ('RSUN_OBS', float),    # Necessary for cutting edges
        ('DSUN_OBS', float)]    # Necessary for limb darkening correction

    keys_ic = isa.sun.get_keys_from_drms(ds_ic, KeyList)
    paths_ic = isa.sun.get_paths_from_drms(ds_ic)
    paths_mag = isa.sun.get_paths_from_drms(ds_mag)

    # Remove the keys and paths of bad quality data points

    date_obs = list(keys_ic.keys())
    remove_index = []

    for day in range(len(keys_ic)):
        if keys_ic[date_obs[day]]["QUALITY"] != "0x00000000":
            remove_index.append(day)
        path = os.path.join(paths_ic[day], 'continuum.fits')
        if not os.path.isfile(path):
            remove_index.append(day)

    unique_index = np.unique(remove_index)
    unique_index = unique_index.tolist()
    unique_index.reverse()

    for index in unique_index:
        x = date_obs.pop(index)
        keys_ic.pop(x)
        paths_ic.pop(index)
        paths_mag.pop(index)

    date_obs = list(keys_ic.keys())

    path = './Datasets_1024x1024/'
    f = open(path + "dates_year_{}_every_day.txt".format(year), "w")
    f.write(str(keys_ic.keys()))    
    f.close()

    n_days = len(date_obs)
    n_pixels = 4096

    list_ic = []
    list_mag = []


    # %% Load data from netDRMS

    print("Initializing script...")

    for day in tqdm(range(n_days)):
        
        a = os.path.join(paths_ic[day], 'continuum.fits')
        b = os.path.join(paths_mag[day], 'magnetogram.fits')
        
        data_ic = fits.getdata(a)
        data_mag = fits.getdata(b)

        
    # %% Apply limb darkening correction and normalization

        data_ic_corrected = isa.sun.limb_darkening_correction(data_ic, keys_ic[date_obs[day]])
        data_ic_normalized = data_ic_corrected - np.nanmean(data_ic_corrected)

        normalization_factor = 2500. # in G
        data_mag_normalized = data_mag / normalization_factor
        data_mag_normalized[np.where(data_mag_normalized > 1.)] = 1.
        data_mag_normalized[np.where(data_mag_normalized < -1.)] = -1.


    # %% Cut edges of data

        x0 = keys_ic[date_obs[day]]["CRPIX1"] - 1
        y0 = keys_ic[date_obs[day]]["CRPIX2"] - 1
        pixel_spacing = keys_ic[date_obs[day]]["CDELT1"]
        r_sun = keys_ic[date_obs[day]]["RSUN_OBS"]

        n_pixels_sun = r_sun / pixel_spacing - 17

        [X, Y] = np.mgrid[0:n_pixels, 0:n_pixels]
        xpr = X - x0
        ypr = Y - y0
        reconstruction_circle = (xpr ** 2 + ypr ** 2) <= n_pixels_sun ** 2
        
        data_ic_cut = 1. * data_ic_normalized
        data_mag_cut = 1. * data_mag_normalized

        data_ic_cut[~reconstruction_circle] = np.nan
        data_mag_cut[~reconstruction_circle] = np.nan


    # %% Degrade data

        new_res = 1024

        data_mag_degraded = isa.sun.degrade_image(data_mag_cut, new_res)
        data_ic_degraded = isa.sun.degrade_image(data_ic_cut, new_res)

        
    # %% Divide dataset in 64 sub-images

        n_slices = 64
        n_pixels_per_row = int(np.sqrt(new_res**2 / n_slices))
        
        data_mag_sliced = []
        data_ic_sliced = []
        
        for x in range(0, new_res, n_pixels_per_row):
            for y in range(0, new_res, n_pixels_per_row):
                tile = data_mag_degraded[x:x+n_pixels_per_row, y:y+n_pixels_per_row]
                data_mag_sliced.append(tile)
                tile = data_ic_degraded[x:x+n_pixels_per_row, y:y+n_pixels_per_row]
                data_ic_sliced.append(tile)

        data_mag_sliced = np.reshape(data_mag_sliced, (64, 128, 128))
        data_ic_sliced = np.reshape(data_ic_sliced, (64, 128, 128))


    # %% Cut the first two and last two rows + images partially outside of solar disc

        data_mag_sliced = data_mag_sliced[16:48]
        data_ic_sliced = data_ic_sliced[16:48]

        n_slices = len(data_mag_sliced)
        boolean_array = np.empty((n_slices), int)

        for slices in range(n_slices):
            boolean_array[slices] = ~np.isnan(data_mag_sliced[slices]).any()

        data_mag = data_mag_sliced[np.where(boolean_array)]
        data_ic = data_ic_sliced[np.where(boolean_array)]

            
    # %% Take only sub-images with spots
        
        n_slices = len(data_mag)
        
        for slices in range(n_slices):
            spot = np.logical_and(data_ic[slices] > -50000, data_ic[slices] <= -15000)
            if spot.any():
                np.save(path + 'mag_slice_{}_year_{}_day_{}'.format(slices, year, day), data_mag[slices])
                np.save(path + 'ic_slice_{}_year_{}_day_{}'.format(slices, year, day), data_ic[slices])

                # list_mag.append(data_mag[slices])
                # list_ic.append(data_ic[slices])


    # %% Save datasets

    # print("Saving datasets...")

    # np.save('./Degraded_Datasets/magnetograms_{}x{}_year_{}_every_day'.format(new_res, new_res, year), list_mag)
    # np.save('./Degraded_Datasets/continuum_{}x{}_year_{}_every_day'.format(new_res, new_res, year), list_ic)

# %% Finish script

print("Finished!")

end = datetime.now()
total_time = end - start
print("Script run time: " + str(total_time))

