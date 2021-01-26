from skimage.io import imread, imsave
from stardist.models import StarDist2D
from csbdeep.utils import Path, normalize
from skimage import morphology, measure
import os, glob, tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stardist import random_label_cmap, _draw_polygons, export_imagej_rois
from stardist.models import StarDist2D

np.random.seed(6)
lbl_cmap = random_label_cmap()

################################
imgFolder = os.path.join('..','2021-01-14_NMGstain_coated_and_uncoated','coated-wt-triton')
outputFolder = os.path.join('..','results_analysis','coated-wt-triton')
################################

# prints a list of available models 
StarDist2D.from_pretrained()
# load the versatile 2D model
model = StarDist2D.from_pretrained('2D_versatile_fluo')

# load image
imgName = glob.glob(os.path.join(imgFolder,'r6*.tif'))[0]
print(imgName)
X = imread(imgName)[8,]
print(X.shape)

n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]
axis_norm = (0,1)

img = normalize(X[1], 1,99.8, axis=axis_norm)
labels, _ = model.predict_instances(img)

plt.figure(figsize=(8,8))
plt.imshow(img if img.ndim==2 else img[...,0], clim=(0,1), cmap='gray')
plt.imshow(labels, cmap=lbl_cmap, alpha=0.5)
plt.axis('off')
plt.show()

labels2 = morphology.remove_small_objects(labels, 1000)
labels2 = measure.label(labels2, connectivity=labels2.ndim)

prop_names = ['label','bbox','centroid','area','perimeter',
            'minor_axis_length','major_axis_length','eccentricity',
            'mean_intensity','min_intensity','max_intensity']
props = pd.DataFrame(measure.regionprops_table(labels2, intensity_image=np.moveaxis(X,0,-1),
                                               properties=prop_names))
imsave(os.path.join(outputFolder,'mask.tif'), labels2.astype(np.uint16))
prop_names_ordered = list(props.keys())
prop_names_ordered.insert(0,'fileName')
props['fileName'] = imgName

props = props[prop_names_ordered]

props.to_csv(os.path.join(outputFolder,'props.csv'))

plt.figure(figsize=(8,8))
plt.imshow(img if img.ndim==2 else img[...,0], clim=(0,.7), cmap='gray')
plt.imshow(labels2, cmap=lbl_cmap, alpha=0.5)
plt.axis('off')
plt.show()
