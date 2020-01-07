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

samples_uv, samples_blu, total_frames, total_tiffs, color_seq = illum_seq(nidaq, tiffdir)

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
def extract_opto_frames(nidaq_opto, wf_samples, length_opto = 10000, length_frame = 62, buffer=400):
    samples_opto = np.nonzero(np.diff(np.int32(nidaq_opto>1.5)) > 0)[0]
    trial_start = samples_opto[1:][np.diff(samples_opto)>1000]
    opto_trials = np.empty((len(trial_start),2), dtype='int64')
    opto_trials[:,0] = trial_start
    opto_trials[:,1] = trial_start + length_opto
    
    opto_blue = []
    for trial in range(len(opto_trials)):    
        opto_blue.append(wf_samples[np.where(np.logical_and(wf_samples>=opto_trials[trial,0], wf_samples<=opto_trials[trial,1]))])
    
    opto_blue_clean = []
    opto_blue_opto = []
    
    for trial in range(len(opto_blue)):
        for frame in opto_blue[trial]:
            if (nidaq_opto[frame-buffer:frame+length_frame] < 1.5).all():
                opto_blue_clean.append(frame)
            else:
                opto_blue_opto.append(frame)
                
    frames_clean = [np.where(samples_blu == i)[0][0] for i in opto_blue_clean]
    frames_opto = [np.where(samples_blu == i)[0][0] for i in opto_blue_opto ]
    
    return frames_clean, frames_opto
#%%
frames_cleanb, frames_optob = extract_opto_frames(nidaq[3,:], samples_blu)

#%%check accuracy of opto frame finder
avg_clean = np.mean(zs_blue_frames[frames_cleanb,:,:], axis = 0)
avg_opto = np.mean(zs_blue_frames[frames_optob,:,:], axis = 0)
plt.imshow(avg_clean, vmin=-3, vmax=3)
plt.figure()
plt.imshow(avg_opto, vmin=-3, vmax=3)
plt.figure()
plt.imshow(avg_clean - avg_opto, vmin=-3, vmax=3)
#%%

frame_starts = np.array(frames_cleanb)[np.diff([0]+frames_cleanb)>50]
#%%
frame_by_frame = np.empty((len(frame_starts), 9, 125, 200))

for frame in frames_cleanb:
    if frame in frame_starts:
        trial = np.argwhere(frame_starts==frame)[0][0]
        frame_by_frame[trial, 0,:,:] = zs_blue_frames[frame,:,:]
    else:
        trial = np.argmin(abs(frame-frame_starts))
        f = frames_cleanb[ frames_cleanb.index(frame_starts[trial]):].index(frame)
        if f<9:
            frame_by_frame[trial, f,:,:] = zs_blue_frames[frame,:,:]
#%%
            
#%%
episodes = np.empty((len(frame_starts), 31), dtype='int64')

for i in range(len(frame_starts)):
    episodes[i,0:15] = np.arange(frame_starts[i]-15, frame_starts[i])
    a = np.where(frames_clean==frame_starts[i])[0][0]
    episodes[i,15:24] = frames_clean[a:a+9]
    t = 24
    while t<31:
        episodes[i,t] = episodes[i, t-1] + 1
        t+=1
#%%
a = np.empty((31,125,200,76),dtype=np.float32)
for i in range(76):
    a[:,:,:,i] = zs_blue_frames[episodes[i],:,:]
#%%
animate_frameseq(b, zmin=-1, zmax=1, filename='test', fps = 4, colormap = 'seismic', savefile=True)