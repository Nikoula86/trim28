import pandas as pd
import os, tqdm, struct
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from skimage.io import imread, imsave

########################################

folderPath = os.path.join('Y:',os.sep,'Nicola_Gritti','raw_data','immuno_NMG','2021-01-14_NMGstain_coated_and_uncoated')
down = 1
overlap = int(79/down)
nrows = 14
ncols = 12
plane = 8

########################################

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

dim1 = int(2160/down*nrows-overlap*(nrows-1))
dim2 = int(2160/down*ncols-overlap*(ncols-1))
imgs_all = np.zeros((4,dim1,dim2)).astype(np.uint16)

folder_names = [
                    os.path.join(folderPath,i) for i in [
                        # 'uncoated_wt-triton',
                        'coated_wt-triton'
                        ]
                ]

for folder in folder_names:
    # for row in np.arange(nrows):
    for col in tqdm.tqdm(np.arange(ncols)):
        for row in np.arange(nrows):
            c1 = int(2160/down/2+row*(2160/down-overlap))
            c2 = int(2160/down/2+col*(2160/down-overlap))

            filename = os.path.join(folder,'r%dc%d.tif'%(row,col))
            img = imread(filename)[8,:4,::-1,::]
            img = img[:,::down,::down]
            # print(img.shape, c1, c2)

            imgs_all[:,int(c1-2160/down/2):int(c1+2160/down/2),
                    int(c2-2160/down/2):int(c2+2160/down/2)] = img

            # create imagej metadata with LUTs
            luts_dict = make_lut()
            luts_name = ['gray','blue','green','orange']
            ijtags = imagej_metadata_tags({'LUTs': [luts_dict[i] for i in luts_name]}, '>')

            # if row == 2:
            #     imsave('stitched_test.tif', imgs_all, byteorder='>', imagej=True,
            #                             metadata={'mode': 'composite'}, extratags=ijtags)

    imsave(folder+'-stitched_%d%d1.tif'%(down,down), imgs_all, byteorder='>', imagej=True,
                            metadata={'mode': 'composite'}, extratags=ijtags)

