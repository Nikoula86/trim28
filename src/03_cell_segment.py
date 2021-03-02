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

pc = os.environ['COMPUTERNAME']
if pc=='PCBA-TRIVEDI03': # my Razer
    folder_raw = os.path.join('E:',os.sep,'immuno_NMG')
elif pc=='PCBA-TRIVEDI02': # workstation
    folder_raw = os.path.join('Y:',os.sep,'Nicola_Gritti','raw_data','immuno_NMG')

exp_folder = os.path.join('2021-02-14_NMG_immuno2_fullSlide2')

imgFolder = os.path.join(folder_raw, exp_folder, 'results')
outputFolder = os.path.join(folder_raw, exp_folder, 'results')

imgNames = [
            'slide2_var653','slide2_var654','slide2_var708','slide2_var709',                        
            'slide2_GFP','slide2_WT','slide2_var180','slide2_var374'

            # 'rep1_GFP-stitched_%d%d.tif',
            # 'rep2_GFP-stitched_%d%d.tif',
            # 'rep1_WT-stitched_%d%d.tif',
            # 'rep2_WT-stitched_%d%d.tif',
            # 'rep3_WT-stitched_%d%d.tif',
            # 'rep4_WT-stitched_%d%d.tif',
            # 'rep5_WT_noSep-stitched_%d%d.tif',
            # 'rep6_WT_noTrim-stitched_%d%d.tif',
            
            # 'rep1_WT-stitched_%d%d.tif',
            # 'rep1_empty-stitched_%d%d.tif',
            # 'rep2_WT-stitched_%d%d.tif',
            # 'rep2_GFP-stitched_%d%d.tif',

            # 'rep1_C01-stitched_%d%d.tif',
            # 'rep1_C02-stitched_%d%d.tif',
            # 'rep1_D01-stitched_%d%d.tif',
            # 'rep1_D02-stitched_%d%d.tif',
            ]
imgNames = [i+'-stitched_%d%d.tif' for i in imgNames]

down = 1

channel_segment = 1 # DAPI
vis_down = 4

make_segmentation = True

################################

def remove_large_objects(segments, max_size):
    out = np.copy(segments)
    component_sizes = np.bincount(segments.ravel())
    too_small = component_sizes > max_size
    too_small_mask = too_small[segments]
    out[too_small_mask] = 0
    return out

################################

# prints a list of available models 
StarDist2D.from_pretrained()
# load the versatile 2D model
model = StarDist2D.from_pretrained('2D_versatile_fluo')

for imgName in imgNames:

    # ##load image
    print('Loading image...')
    imgName = imgName%(down,down)
    X = imread(os.path.join(imgFolder,imgName))
    print('Done.',X.shape)
    
    # normalize and segment cells in the image
    print('Normalizing...')
    axis_norm = (0,1)
    img = normalize(X[:,:,channel_segment], 1,99.8, axis=axis_norm)
    
    if make_segmentation:
    
        # use block_size and n_tiles to avoid OOM
        print('Predicting cells with StarDist...')
        labels, _ = model.predict_instances_big(img, axes='YX', 
                                                    block_size=4096,
                                                    min_overlap=128, 
                                                    n_tiles=(8,8), 
                                                    show_progress=True)
        print('Done.', img.shape)
    
        # plt.figure(figsize=(8,8))
        # plt.imshow(img[::vis_down,::vis_down], clim=(0,1), cmap='gray')
        # plt.imshow(labels[::vis_down,::vis_down], cmap=lbl_cmap, alpha=0.5)
        # plt.axis('off')
        # plt.show()
    
        ### remove small objects, relabel and save mask
        print('Removing small objects and relabeling...')
        labels = morphology.remove_small_objects(labels, 1000)
        labels = remove_large_objects(labels, 50000)
        labels = measure.label(labels, connectivity=labels.ndim)
        print('Saving mask...')
        imsave(os.path.join(outputFolder,'mask_'+imgName), labels.astype(np.uint16))
    
    else:
        print('Reading mask...')
        labels = imread(os.path.join(outputFolder,'mask_'+imgName))
        print('Done.')
    
    ### Find cell props
    # use skimage measure regionprops to extract all features
    print('Extracting region props...')
    prop_names = ['label','bbox','centroid','area','perimeter',
                'minor_axis_length','major_axis_length','eccentricity',
                'mean_intensity','min_intensity','max_intensity']
    props = pd.DataFrame(measure.regionprops_table(labels, intensity_image=X,
                                                   properties=prop_names))
    prop_names_ordered = list(props.keys())
    prop_names_ordered.insert(0,'fileName')
    props['fileName'] = imgName
    props = props[prop_names_ordered]
    print('Saving regionprops in csv...')
    props.to_csv(os.path.join(outputFolder,'props_'+os.path.splitext(imgName)[0]+'.csv'))
    print('Done.')
    
    # plt.figure(figsize=(8,8))
    # plt.imshow(img[::vis_down,::vis_down], clim=(0,.7), cmap='gray')
    # plt.imshow(labels[::vis_down,::vis_down], cmap=lbl_cmap, alpha=0.5)
    # plt.axis('off')
    # plt.show()
