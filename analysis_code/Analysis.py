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

from WF_functions import load_nidaq, illum_seq, get_tif_fr_ilu, load_and_filter, smooth_pool, animate_frameseq, extract_opto_frames, get_target_frames, plot_result
#%%
os.chdir('E:/WF/12.12.2019')

tiffdir = '2019-12-12T10_50_12_WFvlgn02'
nidaqfile = 'nidaq2019-12-12T10_50_07_WFvlgn02.bin'
frames_csv = 'widefield2019-12-12T10_50_10_WFvlgn02.csv'
#%%
nidaq, chan_labels = load_nidaq(nidaqfile)

samples_uv, samples_blu, total_frames, total_tiffs, color_seq = illum_seq(nidaq, tiffdir)

tiffs_frames_color = get_tif_fr_ilu(frames_csv, color_seq)
#%%load blue frames, downsample
blue_frames, files_loaded_blue, frames_loaded_blue = load_and_filter(tot_frames=total_frames, tfc_array=tiffs_frames_color, c=6, colorsequence=color_seq, tiffdir=tiffdir)
ds_blue_frames = smooth_pool(blue_frames, k=5)
blue_frames = None
#%%load uv frames, downsample
uv_frames, files_loaded_uv, frames_loaded_uv = load_and_filter(tot_frames=total_frames, tfc_array=tiffs_frames_color, c=5, colorsequence=color_seq, tiffdir=tiffdir)
ds_uv_frames = smooth_pool(uv_frames, k=5)
uv_frames = None
#%%fill in empty frames
avg_frame = np.mean(ds_blue_frames, axis = 0)
for frame in range(len(ds_blue_frames)):
    if np.sum(ds_blue_frames[frame]) == 0:
        ds_blue_frames[frame] = avg_frame
#%%
avg_frame = np.mean(ds_uv_frames, axis = 0)
for frame in range(len(ds_uv_frames)):
    if np.sum(ds_uv_frames[frame]) == 0:
        ds_uv_frames[frame] = avg_frame
#%%extract opto frames, separate into clean and during pulse
frames_opto_clean_b, frames_opto_pulse_b, opto_trial_start, opto_trial_end, samples_opto_b = extract_opto_frames(nidaq[3,:], samples_blu, buffer = 100)
#frames_opto_clean_uv, frames_opto_pulse_uv, opto_trial_start, opto_trial_end, samples_opto_uv = extract_opto_frames(nidaq[3,:], samples_uv, buffer = 100)
#%%Hemodynamic or normalisation
avg_blue = np.mean(ds_blue_frames, axis = 0)
avg_uv = np.mean(ds_uv_frames, axis = 0)

hemo = np.empty_like(ds_blue_frames)

if color_seq[0]==6:
    hemo[0,:,:] = avg_uv
    hemo[-1,:,:] = avg_uv
    for i in range(1,len(hemo)-1):
        hemo[i,:,:] = np.mean((ds_uv_frames[i-1], ds_uv_frames[i]), axis = 0)
        
if color_seq[0]==5:
    hemo[0,:,:] = np.mean(ds_uv_frames, axis = 0)
    hemo[-1,:,:] = np.mean(ds_uv_frames, axis = 0)
    for i in range(1,len(hemo)-1):
        hemo[i,:,:] = np.mean((ds_uv_frames[i], ds_uv_frames[i+1]), axis = 0)

hemo_corr = (ds_blue_frames/hemo)/(avg_blue/np.mean(hemo, axis=0))
hemo_corr_norm = hemo_corr/np.std(hemo_corr, axis = 0)
#%%
zs_blue = (ds_blue_frames - np.mean(ds_blue_frames, axis = 0))/np.std(ds_blue_frames, axis = 0)
#%%Obtain baselines for each opto trial
bl_frames = np.empty((len(samples_opto_b),100,160), dtype=np.float64)

for i in range(len(bl_frames)):
    last = np.where(samples_blu == samples_opto_b[i][0])[0][0] - 1
    first = last - 12
    bl_frames[i,:,:] = np.mean(zs_blue[first:last,:,:], axis = 0)
#%%Get frames of interest for plotting
frame_1 = get_target_frames(samples_opto_b, 0, samples_blu, frames_opto_clean_b)
frame_2 = get_target_frames(samples_opto_b, 1, samples_blu, frames_opto_clean_b)
frame_3 = get_target_frames(samples_opto_b, 2, samples_blu, frames_opto_clean_b)
frame_last = get_target_frames(samples_opto_b, -1, samples_blu, frames_opto_clean_b)

#%%Plot and save frames of interest
result_frame3 = plot_result('Frame_3 wfVLGN02',frame_3,zs_blue,bl_frames,save='off',x=100,y=160,z=0.6)

#%%Z score by baseline
#bl_frames_raw = np.empty((len(samples_opto_b),12,100,160), dtype=np.float64)

#for i in range(len(bl_frames)):
#    last = np.where(samples_blu == samples_opto_b[i][0])[0][0] - 1
#    first = last - 12
#    bl_frames_raw[i,:,:,:] = ds_blue_frames[first:last,:,:]
#%
#title = '_First_frame_clean'
#result_first_frame_bl = np.empty((len(first_frame), 100,160))
#for i in range(len(first_frame)):
#    result_first_frame_bl[i,:,:]=(ds_blue_frames[first_frame[i]] - np.mean(bl_frames_raw[i,:,:,:], axis =0 ))/ np.std(bl_frames_raw[i,:,:,:], axis=0)
#plt.figure()
#plt.imshow(np.mean(result_first_frame_bl, axis=0), vmin=-1, vmax=1, cmap = 'seismic')
#plt.title(title)
#plt.axis('off')