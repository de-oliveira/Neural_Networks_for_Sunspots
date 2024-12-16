#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 15:05:15 2023

@author: oliveira
"""

import os
from astropy.io import fits
import numpy as np
import IsabelaFunctions as isa
from tqdm import tqdm
from datetime import datetime

f = 14
start = datetime.now()

# which_year = 2011
# full_year = 365
# every_x_days = 2

# ds_ic = 'hmi.Ic_45s[{}.01.02_12:00:00_TAI/{}d@{}d]'.format(which_year, full_year, every_x_days)
# ds_mag = 'hmi.m_45s[{}.01.02_12:00:00_TAI/{}d@{}d]'.format(which_year, full_year, every_x_days)

which_year = 2014
full_year = 365
every_x_hours = 4

ds_ic = 'hmi.Ic_45s[{}.01.02_12:00:00_TAI/{}d@{}h]'.format(which_year, full_year, every_x_hours)
ds_mag = 'hmi.m_45s[{}.01.02_12:00:00_TAI/{}d@{}h]'.format(which_year, full_year, every_x_hours)


# %% Load data from netDRMS

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
    
# Get data from paths
print("Getting data: ")

n_days = len(paths_ic)

f = open("./Degraded_Datasets/dates_year_{}_every_{}_hours.txt".format(which_year, every_x_hours), "w")
f.write(str(keys_ic.keys()))    
f.close()

n_pixels = 4096

data_ic = np.empty((n_days, n_pixels, n_pixels))
data_mag = np.empty_like(data_ic)

for day in tqdm(range(n_days)):
    a = os.path.join(paths_ic[day], 'continuum.fits')
    b = os.path.join(paths_mag[day], 'magnetogram.fits')
    
    data_ic[day, :, :] = fits.getdata(a)
    data_mag[day, :, :] = fits.getdata(b)


# %% Apply limb darkening correction

data_ic_corrected = np.empty_like(data_ic)

print("Applying limb darkening correction: ")
for day in tqdm(range(n_days)):
    data_ic_corrected[day, :, :] = isa.sun.limb_darkening_correction(data_ic[day], keys_ic[date_obs[day]])

del data_ic


# %% Cut edges of data (otherwise there will be a ring of 'spots')

data_ic_corrected_cut = 1. * data_ic_corrected
data_mag_cut = 1. * data_mag

print("Cutting edge of images: ")
for day in tqdm(range(n_days)):
    x0 = keys_ic[date_obs[day]]["CRPIX1"] - 1
    y0 = keys_ic[date_obs[day]]["CRPIX2"] - 1
    pixel_spacing = keys_ic[date_obs[day]]["CDELT1"]
    r_sun = keys_ic[date_obs[day]]["RSUN_OBS"]

    radius_sun = r_sun / pixel_spacing - 17

    [X, Y] = np.mgrid[0:n_pixels, 0:n_pixels]
    xpr = X - x0
    ypr = Y - y0
    reconstruction_circle = (xpr ** 2 + ypr ** 2) <= radius_sun ** 2
    
    data_ic_corrected_cut[day][~reconstruction_circle] = np.nan
    data_mag_cut[day][~reconstruction_circle] = np.nan

del data_ic_corrected, data_mag


# %% Degrading dataset

new_res = 1024
data_mag_degraded = np.empty((n_days, new_res, new_res))
data_ic_degraded = np.empty_like(data_mag_degraded)

print("Degrading data: ")
for day in tqdm(range(n_days)):
    data_mag_degraded[day, :, :] = isa.sun.degrade_image(data_mag_cut[day, :, :], new_res)
    data_ic_degraded[day, :, :] = isa.sun.degrade_image(data_ic_corrected_cut[day, :, :], new_res)

del data_mag_cut, data_ic_corrected_cut


# %% Save dataset

print("Saving datasets...")
np.save('./Degraded_Datasets/magnetograms_1024x1024_year_{}_every_{}_hours'.format(which_year, 
                                                        every_x_hours), data_mag_degraded)
np.save('./Degraded_Datasets/continuum_1024x1024_year_{}_every_{}_hours'.format(which_year, 
                                                        every_x_hours), data_ic_degraded)

print("Finished!")

end = datetime.now()
total_time = end - start
print("Script run time: " + str(total_time))
