"""
Created on Nov 24 2023

@author: oliveira
"""

# %% Import libraries
import torch
from torchvision.transforms import v2
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

path = '/data/seismo/oliveira/Irradiance/Machine_Learning/Large_Dataset/'
folder = 'Datasets_1024x1024/'
new_folder = "Dataset_Augmented/"


# %% Removing old files

for file in glob.glob(path + folder + "*_rot90*"):
    os.remove(file)

for file in glob.glob(path + folder + "*_rot180*"):
    os.remove(file)

for file in glob.glob(path + folder + "*_rot270*"):
    os.remove(file)

for file in glob.glob(path + folder + "*_flipLR*"):
    os.remove(file)

for file in glob.glob(path + folder + "*_flipUD*"):
    os.remove(file)

for file in glob.glob(path + folder + "*_elastic*"):
    os.remove(file)


# %% Select a subset of images with large spots

mask_files = [os.path.split(f)[-1][:-4] 
              for f in glob.glob(path + folder + "mask_*")]

mag_files = [os.path.split(f)[-1][:-4] 
              for f in glob.glob(path + folder + "mag_*")]

# Making sure the files are in the same order
mask_files = sorted(mask_files)
mag_files = sorted(mag_files)

step = 0
for image in range(len(mask_files)):
    a = mask_files[image][7:]
    b = mag_files[image][3:]
    if a == b:
        step += 1
print(step, len(mask_files))


index_large_spots = []
for i in range(len(mask_files)):
    a = np.load(path + folder + mask_files[i] + ".npy")
    if np.count_nonzero(a) >= 1000:
        index_large_spots.append(i)

n_samples = len(index_large_spots)
print(n_samples)

# Print the images
for i in range(n_samples):
    a = np.load(path + folder + mask_files[index_large_spots[i]] + ".npy")
    b = np.load(path + folder + mag_files[index_large_spots[i]] + ".npy")
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(a, cmap='gray')
    ax[1].imshow(b, cmap='gray')
    plt.title("{}".format(mask_files[index_large_spots[i]]))
    plt.show()

""" 
Each image has resolution of 128x128 pixels.
We have a total of 12173 images in the dataset.

- 2001 images with more than 200 spot pixels in the mask
- 415 images with more than 500 spot pixels in the mask (3% of the image)
- 62 images with more than 1000 spot pixels in the mask (6% of the image)
- 14 images with more than 1683 spot pixels in the mask (10% of the image)
"""


n_pixels = []
for i in range(len(index_large_spots)):
    a = np.load(path + folder + mask_files[index_large_spots[i]] + ".npy")
    n_pixels = np.append(n_pixels, np.sum(a))
plt.plot(n_pixels)

max_pixels = np.where(n_pixels == np.max(n_pixels))[0][0]
print(max_pixels)
a = np.load(path + folder + mask_files[index_large_spots[max_pixels]] + ".npy")
plt.imshow(a, cmap='gray')

largest_spot_mask = mask_files[index_large_spots[max_pixels]]
plt.imshow(np.load(path + folder + largest_spot_mask + ".npy"), cmap='gray')

largest_spot_mag = mag_files[index_large_spots[max_pixels]]
plt.imshow(np.load(path + folder + largest_spot_mag + ".npy"), cmap='gray')

"""
The file with the largest spot is 'ic_slice_21_year_2014_day_287'.
This image contains 3589 spot pixels = 22% of the image.
"""


# %% Augmenting one image

a = torch.from_numpy(np.load(path + folder + largest_spot_mask + ".npy"))
plt.imshow(a, cmap='gray')

horizontal_flip = v2.RandomHorizontalFlip(p=1)
a_flipped = horizontal_flip(a)
plt.imshow(a_flipped, cmap='gray')

vertical_flip = v2.RandomVerticalFlip(p=1)
a_flipped2 = vertical_flip(a)
plt.imshow(a_flipped2, cmap='gray')

a_rotated = np.rot90(a)
a_rot90 = torch.from_numpy(a_rotated.copy())

a_rotated = np.rot90(a, 2)
a_rot180 = torch.from_numpy(a_rotated.copy())

a_rotated = np.rot90(a, 3)
a_rot270 = torch.from_numpy(a_rotated.copy())

# v2.ElasticTransform does not work, so we define:
def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    assert len(image.shape)==2

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
    
    return map_coordinates(image, indices, order=1).reshape(shape)

n_elastic = 40
for i in range(n_elastic):
    rs = np.random.RandomState(i)
    alpha = 720
    sigma = 24
    a_elastic = elastic_transform(a, alpha, sigma, rs)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(a, cmap='gray')
    ax[1].imshow(a_elastic, cmap='gray')
    plt.title("Alpha = {}, Sigma = {}".format(alpha, sigma))


# %% Augmenting all images

n_elastic = 40
alpha = 720
sigma = 24

for image in index_large_spots:

    a = np.load(path + folder + mask_files[image] + ".npy")
    np.save(path + new_folder + mask_files[image], a)
    np.save(path + new_folder + mask_files[image] + "_rot90.npy", np.rot90(a))
    np.save(path + new_folder + mask_files[image] + "_rot180.npy", np.rot90(a, 2))
    np.save(path + new_folder + mask_files[image] + "_rot270.npy", np.rot90(a, 3))
    np.save(path + new_folder + mask_files[image] + "_flipLR.npy", np.fliplr(a))
    np.save(path + new_folder + mask_files[image] + "_flipUD.npy", np.flipud(a))
    for n in range(n_elastic):
        rs = np.random.RandomState(n)
        a_elastic = elastic_transform(a, alpha, sigma, rs)
        np.save(path + new_folder + mask_files[image] + "_elastic{}.npy".format(n), a_elastic)
    
    a = np.load(path + folder + mag_files[image] + ".npy")
    np.save(path + new_folder + mag_files[image], a)
    np.save(path + new_folder + mag_files[image] + "_rot90.npy", np.rot90(a))
    np.save(path + new_folder + mag_files[image] + "_rot180.npy", np.rot90(a, 2))
    np.save(path + new_folder + mag_files[image] + "_rot270.npy", np.rot90(a, 3))
    np.save(path + new_folder + mag_files[image] + "_flipLR.npy", np.fliplr(a))
    np.save(path + new_folder + mag_files[image] + "_flipUD.npy", np.flipud(a))
    for n in range(n_elastic):
        rs = np.random.RandomState(n)
        a_elastic = elastic_transform(a, alpha, sigma, rs)
        np.save(path + new_folder + mag_files[image] + "_elastic{}.npy".format(n), a_elastic)

# number of images: 62 * 46 = 2852


# %%
