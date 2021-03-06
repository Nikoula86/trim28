import pandas as pd
import os, tqdm, struct
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from skimage.io import imread, imsave

########################################

pc = os.environ['COMPUTERNAME']
if pc=='PCBA-TRIVEDI03': # my Razer
    folder_raw = os.path.join('E:',os.sep,'immuno_NMG')
elif pc=='PCBA-TRIVEDI02': # workstation
    folder_raw = os.path.join('Y:',os.sep,'Nicola_Gritti','raw_data','immuno_NMG')

exp_folder = os.path.join('2021-02-14_NMG_immuno2_fullSlide2')

folderPath = os.path.join(folder_raw, exp_folder)

down = 1
overlap = int(78/down)
nrows = 14
ncols = 14
planes = [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2
        ]

folder_names = [
                    os.path.join(folderPath,i) for i in [
                        # 'rep1_A01',
                        # 'rep1_A02',
                        # 'rep1_B01',
                        # 'rep1_B02',

                        # 'rep1_GFP',
                        # 'rep2_GFP',
                        # 'rep1_WT',
                        # 'rep2_WT',
                        # 'rep3_WT',
                        # 'rep4_WT',
                        # 'rep5_WT_noSep',
                        # 'rep6_WT_noTrim',

                        'slide2_var653','slide2_var654','slide2_var708','slide2_var709',
                        'slide2_GFP','slide2_WT','slide2_var180','slide2_var374'
                        ]
                ]

n_ch = [
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
        ]

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

i = 1
for folder, plane in zip(folder_names, planes):
    print('%d/%d:'%(i,len(folder_names)), folder)
    well = folder.split('\\')[-1]

    imgs_all = np.zeros((n_ch[i-1],dim1,dim2)).astype(np.uint16)
    # create imagej metadata with LUTs
    luts_dict = make_lut()
    luts_name = ['gray','blue','green','orange']
    ijtags = imagej_metadata_tags({'LUTs': [luts_dict[i] for i in luts_name]}, '>')
    # for row in np.arange(nrows):
    for col in tqdm.tqdm(np.arange(ncols)):
        for row in np.arange(nrows):
            c1 = int(2160/down/2+row*(2160/down-overlap))
            c2 = int(2160/down/2+col*(2160/down-overlap))

            filename = os.path.join(folder,'r%dc%d.tif'%(row,col))
            
            if os.path.exists(filename):
                img = imread(filename)
                if img.ndim==3: ## if 2D image (Y,X,C) expand to fake 3D
                    img = np.expand_dims(img,0)
                # print(img.shape)
                # select best focused plane
                img = img[plane,::-1,::,:]
                # move channel axis in front
                img = np.moveaxis(img,-1,0)
                # downsize image
                img = img[:,::down,::down]
                # print(img.shape, c1, c2)
    
                imgs_all[:,int(c1-2160/down/2):int(c1+2160/down/2),
                        int(c2-2160/down/2):int(c2+2160/down/2)] = img

        # imsave(os.path.join(folderPath,'results',well+'-stitched_%d%d.tif'%(down,down)), imgs_all, byteorder='>', imagej=True,
        #                         metadata={'mode': 'composite'}, extratags=ijtags)


    if not os.path.exists(os.path.join(folderPath,'results')):
        os.mkdir(os.path.join(folderPath,'results'))

    imsave(os.path.join(folderPath,'results',well+'-stitched_%d%d.tif'%(down,down)), imgs_all, byteorder='>', imagej=True,
                            metadata={'mode': 'composite'}, extratags=ijtags, check_contrast=False)

    i += 1

