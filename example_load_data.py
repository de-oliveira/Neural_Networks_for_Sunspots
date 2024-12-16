
# %% Import libraries
import numpy as np
import matplotlib.pyplot as plt

folder = 'Datasets_1024x1024/'


# %% Load data

ic_data = np.load(folder + 'ic_slice_18_year_2014_day_317.npy')
mag_data = np.load(folder + 'mag_slice_18_year_2014_day_317.npy')
ic_mask = np.load(folder + 'mask_ic_slice_18_year_2014_day_317.npy')


# %% Plot data

spot_max = -15000

fig, ax = plt.subplots(1, 3, figsize=(12, 6))
plt.subplots_adjust(hspace = 0., wspace = 0.1)

ax1 = plt.subplot(131)
im1 = plt.imshow(mag_data, cmap = 'gray')
#plt.contour(ic_mask, levels = [0.5], colors = 'red')
plt.contour(ic_data, levels = [-50000, spot_max], colors = 'red')
plt.xticks([])
plt.yticks([])
plt.title('Magnetogram')

ax2 = plt.subplot(132)
im2 = plt.imshow(ic_data, cmap = 'gist_heat')
plt.contour(ic_data, levels = [-50000, spot_max], colors = 'blue')
plt.xticks([])
plt.yticks([])
plt.title('Intensitygram')

ax3 = plt.subplot(133)
im3 = plt.imshow(ic_mask, cmap = 'gray')
plt.contour(ic_data, levels = [-50000, spot_max], colors = 'yellow')
plt.xticks([])
plt.yticks([])
plt.title('Mask')

plt.subplots_adjust(bottom=0.1)
plt.colorbar(im1, ax=ax1, orientation='horizontal')
plt.colorbar(im2, ax=ax2, orientation='horizontal')
plt.colorbar(im3, ax=ax3, orientation='horizontal')

plt.show()
# %%
