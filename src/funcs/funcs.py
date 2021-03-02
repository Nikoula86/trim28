import pandas as pd
import tqdm, os
import numpy as np

def filter_close2supertrans(props):

    transfected = props[props.transfected]
    supertransfected = props[props.supertransfected]

    transfected['dist'] = 0.

    ps = np.array([supertransfected['centroid-0'], supertransfected['centroid-1']])


    for i, cell in transfected.iterrows():
        p = np.array([cell['centroid-0'], cell['centroid-1']])


    print(p,ps.shape)


