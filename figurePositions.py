import pandas as pd
import os, tqdm, struct
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from skimage.io import imread, imsave

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

df = pd.read_csv("metadata.csv")
print(df)

slides = list(set(df.col))
slides.sort()

slide_names = ['uncoated_', 'coated_']

for slide, slide_name in zip(slides, slide_names):
    df_slide = df[df.col==slide]

    xpos = list(set(df_slide.Xpos))
    xpos.sort()
    ypos = list(set(df_slide.Ypos))
    xpos.sort()

    xlims = [np.min(xpos),
            (np.min(xpos)+np.max(xpos))/2,
            np.max(xpos)]

    ylims = [np.min(ypos),
            (np.min(ypos)+(np.min(ypos)+np.max(ypos))/2)/2,
            (np.min(ypos)+np.max(ypos))/2,
            (np.max(ypos)+(np.min(ypos)+np.max(ypos))/2)/2,
            np.max(ypos)]

    # fig, ax = plt.subplots(1,1)
    # ax.scatter(df_slide[df_slide.channel==1].Xpos, df_slide[df_slide.channel==1].Ypos, c=df_slide[df_slide.channel==1]['Unnamed: 0'], s=10)
    # output = np.array(list(product(xlims, ylims)))
    # ax.plot(output[:,0], output[:,1],'or')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # plt.show()

    folderNames = [['wt-triton','empty-triton','wt-triton-mem','empty-triton-mem'],
                   ['wt-tween','empty-tween','wt-tween-mem','empty-tween-mem']]
    for i in range(len(xlims)-1):
        for j in range(len(ylims)-1):

            print(i,j)
            folderName = slide_name+folderNames[i][j]
            if not os.path.exists(folderName):
                os.mkdir(folderName)

            xmin = xlims[i]
            xmax = xlims[i+1]
            ymin = ylims[j]
            ymax = ylims[j+1]

            df_poc = df_slide[(df_slide.Xpos<=xmax)&(df_slide.Xpos>=xmin)]
            df_poc = df_poc[(df_poc.Ypos<=ymax)&(df_poc.Ypos>=ymin)]

            # fig, ax = plt.subplots(1,1)
            # ax.set_title(folderName)
            # ax.scatter(df_poc[df_poc.channel==1].Xpos, df_poc[df_poc.channel==1].Ypos, c=df_poc[df_poc.channel==1]['Unnamed: 0'], s=10)
            # ax.plot(output[:,0], output[:,1],'or')
            # ax.set_xlabel('X')
            # ax.set_ylabel('Y')
            # plt.show()

            xpos_poc = list(set(df_poc.Xpos))
            xpos_poc.sort()
            ypos_poc = list(set(df_poc.Ypos))
            ypos_poc.sort()

            # df_poc_ordered = df_poc.sort_values(by=["Xpos","Ypos"])
            # print(df_poc_ordered)

            for xidx, x in enumerate(xpos_poc):
                for yidx, y in enumerate(ypos_poc):
                    df_pos = df_poc[(df_poc.Xpos==x)&(df_poc.Ypos==y)]
                    channel_list = [2,1,5,3,4]
                    stack = []
                    for ch in tqdm.tqdm(channel_list):
                        df_pos_ch = df_pos[df_pos.channel==ch]
                        df_pos_ch = df_pos_ch.sort_values(by='Zpos')
                        # [print(img_file) for img_file in df_pos_ch.filename]
                        stack_ch = np.stack([imread(os.path.join('Images',img_file)) for img_file in df_pos_ch.filename])
                        stack.append(stack_ch)

                    # order channels
                    stacks = np.array(stack).astype(np.uint16)
                    stacks = np.swapaxes(stacks, 0, 1)

                    # create imagej metadata with LUTs
                    luts_dict = make_lut()
                    luts_name = ['gray','blue','green','orange','magenta']
                    ijtags = imagej_metadata_tags({'LUTs': [luts_dict[i] for i in luts_name]}, '>')

                    outname = 'r%dc%d.tif'%(yidx,xidx)
                    imsave(os.path.join(folderName,outname),stacks, byteorder='>', imagej=True,
                                    metadata={'mode': 'composite'}, extratags=ijtags)


