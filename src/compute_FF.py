import pandas as pd
import os, tqdm, struct, glob
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from skimage.io import imread, imsave

####################################

pc = os.environ['COMPUTERNAME']
if pc=='PCBA-TRIVEDI03': # my Razer
    folder_raw = os.path.join('D:',os.sep,'immuno_NMG')
elif pc=='PCBA-TRIVEDI02': # workstation
    folder_raw = os.path.join('Y:',os.sep,'Nicola_Gritti','raw_data','immuno_NMG')
folder_raw = os.path.join('D:',os.sep,'immuno_NMG')

exp_folder = os.path.join('2021-03-02_FlatField')

slide_names = ['blue','green','orange','red']

for i, slide_name in tqdm.tqdm(enumerate(slide_names)):
    flist = glob.glob(os.path.join(folder_raw, exp_folder,slide_name,'*.tif'))
    imgs = np.stack([imread(f)[:,:,i] for f in flist])
    print(imgs.shape)

    img = np.mean(imgs,0).astype(np.uint16)
    imsave(os.path.join(folder_raw, exp_folder,'FF_%s.tif'%slide_name), img)


