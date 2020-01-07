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
#%%Obtain opto trial starts and ends
samples_opto = np.nonzero(np.diff(np.int32(nidaq[3,:]>1.5)) > 0)[0]
trial_start = samples_opto[1:][np.diff(samples_opto)>1000]
opto_trials = np.empty((len(trial_start),2), dtype='int64')
opto_trials[:,0] = trial_start
opto_trials[:,1] = trial_start + 9500
#%%
opto_blue = []

for trial in range(len(opto_trials)):    
    opto_blue.append(samples_blu[np.where(np.logical_and(samples_blu>=opto_trials[trial,0], samples_blu<=opto_trials[trial,1]))])
#%%

#%%check accuracy of opto trial extraction
plt.scatter(opto_trials[:,0], 1.01*np.ones(len(opto_trials[:,0])), c = 'g')
plt.scatter(opto_trials[:,1], 1.01*np.ones(len(opto_trials[:,0])), c = 'r')
plt.scatter(opto_on, np.ones((len(opto_on)) ), c = 'orange', alpha = 0.2)
plt.plot(nidaq[3,:]/4, alpha = 0.4)