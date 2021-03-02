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

exp_folder = os.path.join('2021-02-21_NMG_rep2_Slide1')

folders = ['testFF']

##################################
def imagej_metadata_tags(metadata, byteorder):
    """Return IJMetadata and IJMetadataByteCounts tags from metadata dict.

    The tags can be passed to the TiffWriter.save function as extratags.

    """
    header = [{'>': b'IJIJ', '<': b'JIJI'}[byteorder]]
    bytecounts = [0]
    body = []

    def writestring(data, byteorder):
        return data.encode('utf-16' + {'>': 'be', '<': 'le'}[byteorder])

    def writedoubles(data, byteorder):
        return struct.pack(byteorder+('d' * len(data)), *data)

    def writebytes(data, byteorder):
        return data.tobytes()

    metadata_types = (
        ('Info', b'info', 1, writestring),
        ('Labels', b'labl', None, writestring),
        ('Ranges', b'rang', 1, writedoubles),
        ('LUTs', b'luts', None, writebytes),
        ('Plot', b'plot', 1, writebytes),
        ('ROI', b'roi ', 1, writebytes),
        ('Overlays', b'over', None, writebytes))

    for key, mtype, count, func in metadata_types:
        if key not in metadata:
            continue
        if byteorder == '<':
            mtype = mtype[::-1]
        values = metadata[key]
        if count is None:
            count = len(values)
        else:
            values = [values]
        header.append(mtype + struct.pack(byteorder+'I', count))
        for value in values:
            data = func(value, byteorder)
            body.append(data)
            bytecounts.append(len(data))

    body = b''.join(body)
    header = b''.join(header)
    data = header + body
    bytecounts[0] = len(header)
    bytecounts = struct.pack(byteorder+('I' * len(bytecounts)), *bytecounts)
    return ((50839, 'B', len(data), data, True),
            (50838, 'I', len(bytecounts)//4, bytecounts, True))

def make_lut():
    # generate LUT for primary and secondary colors

    # Intensity value range
    val_range = np.arange(256, dtype=np.uint8)
    luts_dict = {}
    # Gray LUT
    luts_dict['gray'] = np.stack([val_range, val_range, val_range])
    # Red LUT
    luts_dict['red'] = np.zeros((3, 256), dtype=np.uint8)
    luts_dict['red'][0, :] = val_range
    # Green LUT
    luts_dict['green'] = np.zeros((3, 256), dtype=np.uint8)
    luts_dict['green'][1, :] = val_range
    # Blue LUT
    luts_dict['blue'] = np.zeros((3, 256), dtype=np.uint8)
    luts_dict['blue'][2, :] = val_range
    # Magenta LUT
    luts_dict['magenta'] = np.zeros((3, 256), dtype=np.uint8)
    luts_dict['magenta'][0, :] = val_range
    luts_dict['magenta'][2, :] = val_range
    # Cyan LUT
    luts_dict['cyan'] = np.zeros((3, 256), dtype=np.uint8)
    luts_dict['cyan'][1, :] = val_range
    luts_dict['cyan'][2, :] = val_range
    # Yellow LUT
    luts_dict['yellow'] = np.zeros((3, 256), dtype=np.uint8)
    luts_dict['yellow'][0, :] = val_range
    luts_dict['yellow'][1, :] = val_range
    # Orange LUT
    luts_dict['orange'] = np.zeros((3, 256), dtype=np.uint8)
    luts_dict['orange'][0, :] = val_range
    luts_dict['orange'][1, :] = (165.*val_range/256.).astype(np.uint8)
    # Maroon LUT
    luts_dict['maroon'] = np.zeros((3, 256), dtype=np.uint8)
    luts_dict['maroon'][0, :] = (128.*val_range/256.).astype(np.uint8)

    return luts_dict

######################

fflist = glob.glob(os.path.join('D',os.sep,'immuno_NMG','2021-03-02_FlatField','*.tif'))
fflist.sort()
ffs = np.stack([imread(f) for f in fflist]).astype(float)
ffs = np.stack([ff/np.mean(ff) for ff in ffs])

for folder in folders:
    flist = glob.glob(os.path.join(folder_raw, exp_folder, folder, '*.tif'))
    flist.sort()

    for f in tqdm.tqdm(flist):
        img = imread(f).astype(float)
        print(img.shape)
        if img.ndim==3:
            img = np.expand_dims(img,0)

        # plt.figure()
        # plt.imshow(img[0,:,:,0])
        # plt.figure()
        # plt.imshow(img[0,:,:,1])
        # plt.figure()
        # plt.imshow(img[1,:,:,0])
        # plt.show()

        for ch in [1,2,3]:
            for i in range(img.shape[0]):
                img[i,:,:,ch] = img[i,:,:,ch]*ffs[ch]
        img = img.astype(np.uint16)

        img = np.moveaxis(img,-1,1)
        # img = np.swapaxes(img,0,1)
        print(img.shape)

        # create imagej metadata with LUTs
        luts_dict = make_lut()
        luts_name = ['gray','blue','green','orange']
        ijtags = imagej_metadata_tags({'LUTs': [luts_dict[i] for i in luts_name]}, '>')

        outname = f[:-4]+'_1.tif'
        # print(outname)
        imsave(outname,img, byteorder='>', imagej=True,
                        metadata={'mode': 'composite'}, extratags=ijtags, check_contrast=False)

        # imsave(outname,img)
