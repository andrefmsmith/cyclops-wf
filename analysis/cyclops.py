# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

def loadcyclops(fname):
    data = np.fromfile(fname,dtype=np.float64)
    channels = data.reshape((-1,10))
    uv = np.diff(np.int32(channels[:,5] > 1.5)) > 0
    bl = np.diff(np.int32(channels[:,6] > 1.5)) > 0
    uvsamples = np.nonzero(uv)[0]
    blsamples = np.nonzero(bl)[0]
    return channels,uvsamples,blsamples