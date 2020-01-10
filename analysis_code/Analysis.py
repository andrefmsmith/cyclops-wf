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

from WF_functions import load_nidaq, illum_seq, get_tif_fr_ilu, load_and_filter, smooth_pool, animate_frameseq, extract_opto_frames
#%%
os.chdir('E:/WF/11.12.2019')

tiffdir = '2019-12-11T12_47_01_WFSC02'
nidaqfile = 'nidaq2019-12-11T12_47_09_WFSC02.bin'
frames_csv = 'widefield2019-12-11T12_46_59_WFSC02.csv'
#%%
def illum_seq(nidaq, tiff_folder, u=5, b=6):
    '''1. Obtains UV and Blue illumination onsets from nidaq.
    2. Calculates total_frames and total_tiffs captured.
    3. Establishes the illumination sequence sent during the experiment.
    4 Returns this sequence, UV and blue sample onsets, total_frames and total_tiffs captured.'''
    
    uv = np.diff(np.int32(nidaq[u,:]>1.5)) > 0
    blu = np.diff(np.int32(nidaq[b,:]>1.5)) > 0
    samples_uv = np.nonzero(uv)[0]
    samples_blu = np.nonzero(blu)[0]
    total_frames = len(samples_uv) + len(samples_blu)
    total_tiffs = len(os.listdir(tiff_folder))
    
    color_seq = []
    if samples_uv[0]<samples_blu[0]:
        color_seq.append(u)
    else:
        color_seq.append(b)
    while len(color_seq)<total_frames:
        if color_seq[-1]==u:
            color_seq.append(b)
        else:
            color_seq.append(u)
        
    if sum(np.diff(abs(np.diff(color_seq)))) == 0:
        print('Illumination sequence is as predicted.')
    else:
        print('Check illumination sequence.')
            
    return samples_uv+1, samples_blu+1, total_frames, total_tiffs, color_seq
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

avg_frame = np.mean(ds_uv_frames, axis = 0)
for frame in range(len(ds_uv_frames)):
    if np.sum(ds_uv_frames[frame]) == 0:
        ds_uv_frames[frame] = avg_frame
#%%extract opto frames, separate into clean and during pulse
frames_opto_clean_b, frames_opto_pulse_b, opto_trial_start, opto_trial_end, samples_opto_b = extract_opto_frames(nidaq[3,:], samples_blu, buffer = 100)
frames_opto_clean_uv, frames_opto_pulse_uv, opto_trial_start, opto_trial_end, samples_opto_uv = extract_opto_frames(nidaq[3,:], samples_uv, buffer = 100)
#%%Find right hemodynamic correction frames, correct and normalise to stdev
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
#%%Obtain baselines for each opto trial
bl_frames = np.empty((len(samples_opto_b),100,160), dtype=np.float64)

for i in range(len(bl_frames)):
    last = np.where(samples_blu == samples_opto_b[0][0])[0][0] - 1
    first = last - 12
    bl_frames[i,:,:] = np.mean(hemo_corr_norm[first:last,:,:], axis = 0)
#%%Find frames of interest within clean frames
frames_opto_b = [[]]*len(samples_opto_b)
temp = []

trial = 0
while trial < len(frames_opto_b):
    for sample in samples_opto_b[trial]:
        temp.append( np.where(samples_blu==sample)[0][0])
    frames_opto_b[trial] = temp
    trial +=1
    temp = []
#%
temp = []
first_frame = []
second_frame = []
last_frame = []
trial = 0
while trial < len(frames_opto_b):
    for frame in frames_opto_b[trial]:
        if frame in frames_opto_clean_b:
            temp.append(frame)
    first_frame.append((temp[0]))
    second_frame.append((temp[1]))
    last_frame.append((temp[-1]))
    temp = []
    trial += 1
#%%Save output for frames of interest
title = '_First_frame_clean'
result_first_frame = np.empty((len(first_frame), 100,160))
for i in range(len(first_frame)):
    result_first_frame[i,:,:]=hemo_corr_norm[first_frame[i]] - bl_frames[i]
plt.figure()
plt.imshow(np.mean(result_first_frame, axis=0), vmin=-5, vmax=5, cmap = 'seismic')
plt.title(title)
plt.axis('off')

plt.savefig(tiffdir+title+'.png')
plt.close()
#%%
title = '_Second_frame_clean'
result_second_frame = np.empty((len(second_frame), 100,160))
for i in range(len(first_frame)):
    result_second_frame[i,:,:]=hemo_corr_norm[second_frame[i]] - bl_frames[i]
plt.figure()
plt.imshow(np.mean(result_second_frame, axis=0), vmin=-5, vmax=5, cmap='seismic')
plt.title(title)
plt.axis('off')

plt.savefig(tiffdir+title+'.png')
plt.close()
#%%
title = '_Last_frame_clean'
result_last_frame = np.empty((len(last_frame), 100,160))
for i in range(len(first_frame)):
    result_last_frame[i,:,:]=hemo_corr_norm[last_frame[i]] - bl_frames[i]
plt.figure()
plt.imshow(np.mean(result_last_frame, axis=0), vmin=-5, vmax=5, cmap = 'seismic')
plt.title(title)
plt.axis('off')

plt.savefig(tiffdir+title+'.png')
plt.close()
#%%
title = '_Average_allframes_clean'
result_avg = np.empty_like(bl_frames)
for i in range(len(bl_frames)):
    result_avg[i,:,:] = np.mean(hemo_corr_norm[frames_opto_b[i]], axis = 0) - bl_frames[i]
plt.figure()
plt.imshow(np.mean(result_avg, axis=0), cmap='seismic', vmin=-5, vmax=5)
plt.title(title)
plt.axis('off')

plt.savefig(tiffdir+title+'.png')
plt.close()
#%%
