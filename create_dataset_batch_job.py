#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 15:05:15 2023

@author: oliveira
"""

import os
from astropy.io import fits
import numpy as np
from tqdm import tqdm
from datetime import datetime
import subprocess as subp
from scipy.interpolate import RegularGridInterpolator


def get_keys_from_drms(ds, keylist):
    
    knames = ','.join([nam for nam, typ in keylist])
    p = subp.Popen('show_info ds=%s key=%s -q' % (ds, knames),
                    shell=True, stdout=subp.PIPE, encoding='utf-8')
    lines = [line.rstrip() for line in p.stdout.readlines()]
    keys_str = np.array([line.split() for line in lines])
    keys = {}
    for i in range(keys_str.shape[0]):
        line = keys_str[i]
        keys[line[0]] = {}
        for j in range(1, keys_str.shape[1]):
            keys[line[0]][keylist[j][0]] = keylist[j][1](line[j])
    return keys


def get_paths_from_drms(ds):
    p = subp.Popen('show_info ds=%s -Pq' % (ds), shell=True,
                    stdout=subp.PIPE, encoding='utf-8')
    paths = [line.strip() for line in p.stdout.readlines()]
    return paths


def degrade_image(data, new_res):
            
    m = data.shape[0]
    y = np.linspace(0, 1.0/m, data.shape[0])
    x = np.linspace(0, 1.0/m, data.shape[1])
    interpolating_function = RegularGridInterpolator((y, x), data)

    yv, xv = np.meshgrid(np.linspace(0, 1.0/m, new_res), np.linspace(0, 1.0/m, new_res))

    return interpolating_function((xv, yv))


def limb_darkening_correction(data_hmi, keys):

    x0     = float(keys['CRPIX1'])
    y0     = float(keys['CRPIX2'])
    DSUN   = float(keys['DSUN_OBS'])
    dx     = float(keys['CDELT1'])
    
    RSUN = 696e6

    x_raw = (np.arange(np.shape(data_hmi)[0]) + 1 - x0) # + 1 due to wcs
    y_raw = (np.arange(np.shape(data_hmi)[1]) + 1 - y0) # + 1 due to wcs

    xm_raw, ym_raw               = np.meshgrid(x_raw, y_raw)
    key_secant                   = {}
    key_secant["xm"]             = xm_raw
    key_secant["ym"]             = ym_raw
    key_secant["rad_sun_angle"]  = np.arcsin(RSUN / DSUN)
    key_secant["pix_sep_radian"] = np.deg2rad(dx) / 3600.
    key_secant["sun_rad_maj"]    = key_secant["rad_sun_angle"] / key_secant["pix_sep_radian"] 
    key_secant["sun_rad_min"]    = key_secant["sun_rad_maj"] 
    key_secant["maj_ax_ang"]     = 0 # angle of major axis, here assume to be zero. 
    key_secant["secant_thresh"]  = 4 
    
    LD_coeffs = [1.0, 0.429634631, 0.071182463, -0.02522375, -0.012669259, -0.001446241] 

    u = np.sin(key_secant["maj_ax_ang"])
    v = np.cos(key_secant["maj_ax_ang"])
    xm = key_secant["xm"]
    ym = key_secant["ym"]
    maj_ax_proj = (u * xm + v * ym) / key_secant["sun_rad_maj"]
    min_ax_proj = (v * xm - u * ym) / key_secant["sun_rad_min"]
    rho2 = maj_ax_proj ** 2 + min_ax_proj ** 2

    smap = np.zeros(np.shape(rho2))

    mu = np.sqrt(1.0 - rho2[rho2 < 1]) 
    xi = np.log(mu)
    zt = 1.0
    ld = 1.0
    
    for ord in np.arange(1, 6):
        zt *= xi
        ld += LD_coeffs[ord] * zt
    
    smap[rho2 < 1] = 1. / ld
    
    data_corrected = data_hmi * smap
    
    return data_corrected



# %% Initializing script

f = 14
start = datetime.now()
path_seismo = "/data/seismo/oliveira/Irradiance/Machine_Learning/Spot_Masks/"


which_year = 2014
full_year = 365//5
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

keys_ic = get_keys_from_drms(ds_ic, KeyList)
paths_ic = get_paths_from_drms(ds_ic)
paths_mag = get_paths_from_drms(ds_mag)

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

f = open(path_seismo + "Degraded_Datasets/dates_year_{}_every_{}_hours.txt".format(which_year, every_x_hours), "w")
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
    data_ic_corrected[day, :, :] = limb_darkening_correction(data_ic[day], keys_ic[date_obs[day]])

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
    data_mag_degraded[day, :, :] = degrade_image(data_mag_cut[day, :, :], new_res)
    data_ic_degraded[day, :, :] = degrade_image(data_ic_corrected_cut[day, :, :], new_res)

del data_mag_cut, data_ic_corrected_cut


# %% Save dataset

print("Saving datasets...")
np.save(path_seismo + 'Degraded_Datasets/magnetograms_1024x1024_year_{}_every_{}_hours_1'.format(which_year, 
                                                        every_x_hours), data_mag_degraded)
np.save(path_seismo + 'Degraded_Datasets/continuum_1024x1024_year_{}_every_{}_hours_1'.format(which_year, 
                                                        every_x_hours), data_ic_degraded)

print("Finished!")

end = datetime.now()
total_time = end - start
print("Script run time: " + str(total_time))
