# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import tifffile
import imageio
import skimage.measure
from matplotlib import animation
import sys

sys.path.insert(1, 'E:/WF/cyclops-wf/analysis_code/')

from WF_functions import load_nidaq, illum_seq, get_tif_fr_ilu, load_frames, smooth_pool, animate_frameseq
#%%
os.chdir('E:/WF/11.12.2019')

tiffdir = '2019-12-11T13_52_25_WFSC01'
nidaqfile = 'nidaq2019-12-11T13_52_14_WFSC01.bin'
frames_csv = 'widefield2019-12-11T13_52_23_WFSC01.csv'
#%%
nidaq, chan_labels = load_nidaq(nidaqfile)

samples_uv, samples_blu, total_frames, total_tiffs, color_seq = illum_seq(nidaq)

tiffs_frames_color = get_tif_fr_ilu(frames_csv, color_seq)
#%%
blue_frames, files_loaded = load_frames(samples_blu, tiffs_frames_color, color_seq, tiffdir, c=6)

#%%
ds_blue_frames = smooth_pool(blue_frames)
blue_frames = None
#np.save('widefield2019-12-11T13_52_23_WFSC01', ds_blue_frames)
#%%z-score - turn into function
zs_blue_frames = (ds_blue_frames - np.mean(ds_blue_frames, axis = 0)) / np.std(ds_blue_frames, axis = 0)
    
animate_frameseq(zs_blue_frames[0:1000,:,:], zmin = -3, zmax = 3, filename='full_test')