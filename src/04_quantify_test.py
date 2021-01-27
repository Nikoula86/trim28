# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 17:35:14 2021

@author: nicol
"""

from skimage.io import imread
from stardist.models import StarDist2D
from csbdeep.utils import Path, normalize
from skimage import morphology, measure
import os, glob, tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stardist import random_label_cmap, _draw_polygons, export_imagej_rois
from stardist.models import StarDist2D
from matplotlib import rc
rc('font', size=12)
rc('font', family='Arial')
# rc('font', serif='Times')
rc('pdf', fonttype=42)
# rc('text', usetex=True)

np.random.seed(6)
lbl_cmap = random_label_cmap()
colors = ['black','blue','green','orange','maroon']
channels = ['Bright field','DAPI','Trim28','Sepdb1','Membrane']

################################
imgFolder = os.path.join('..','2021-01-14_NMGstain_coated_and_uncoated','coated-wt-triton')
outputFolder = os.path.join('..','results_analysis','coated-wt-triton')
################################

props = pd.read_csv(os.path.join(outputFolder,'props.csv'))
nch = len([i for i in props.keys() if 'mean_intens' in i])

# load image
imgName = glob.glob(os.path.join(imgFolder,'r*.tif'))[0]
print(imgName)
X = imread(imgName)[8,]
print(X.shape)

##################################################
# plot of distributions
fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(5,8))
fig.subplots_adjust(top=0.99, hspace=0.5, bottom=0.1, right=0.99)
for i in range(nch):
    data = props['mean_intensity-%d'%i]
    ax[i].hist(data, bins=10, color=colors[i])
    ax[i].set_xlabel('Channel %d (%s)'%(i,channels[i]))
    ax[i].set_ylabel('Num cells')
plt.show()
