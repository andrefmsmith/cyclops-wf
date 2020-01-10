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

tiffdir = '2019-12-11T13_52_25_WFSC01'
nidaqfile = 'nidaq2019-12-11T13_52_14_WFSC01.bin'
frames_csv = 'widefield2019-12-11T13_52_23_WFSC01.csv'
#%%
nidaq, chan_labels = load_nidaq(nidaqfile)

samples_uv, samples_blu, total_frames, total_tiffs, color_seq = illum_seq(nidaq, tiffdir)

tiffs_frames_color = get_tif_fr_ilu(frames_csv, color_seq)
#%%load blue frames, downsample
blue_frames, files_loaded_blue, frames_loaded_blue = load_and_filter(tot_frames=total_frames, tfc_array=tiffs_frames_color, c=6, colorsequence=color_seq, tiffdir=tiffdir)
ds_blue_frames = smooth_pool(blue_frames, k=5)
blue_frames = None
#load uv frames, downsample
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
frames_opto_clean_b, frames_opto_pulse_b, opto_trial_start, opto_trial_end, samples_opto_b = extract_opto_frames(nidaq[3,:], samples_blu, buffer = 50)
frames_opto_clean_uv, frames_opto_pulse_uv, opto_trial_start, opto_trial_end, samples_opto_uv = extract_opto_frames(nidaq[3,:], samples_uv, buffer = 50)
#%%test opto samples - working with buffer of 50 samples
for sample in samples_opto_b[5]:
    frame = np.where(samples_blu==sample)[0][0]
    if  frame in frames_opto_clean_b:
        plt.figure()
        plt.imshow(ds_blue_frames[frame])
        


#%%create baselines for opto trials
no_opto_frames = [np.where(samples_blu==i)[0][0] for i in samples_blu if nidaq[3,i]<=1.5 ]

ds_blue_frames_norm = ds_blue_frames/np.std(ds_blue_frames[no_opto_frames], axis = 0)

bl_frames = np.empty((len(samples_opto_b),100,160), dtype=np.float64)

for i in range(len(bl_frames)):
    last = np.where(samples_blu == samples_opto_b[0][0])[0][0] - 1
    first = last - 12
    bl_frames[i,:,:] = np.mean(ds_blue_frames_norm[first:last,:,:], axis = 0)
#%%
frames_opto_b = [[]]*len(samples_opto_b)
temp = []

trial = 0
while trial < len(frames_opto_b):
    for sample in samples_opto_b[trial]:
        temp.append( np.where(samples_blu==sample)[0][0])
    frames_opto_b[trial] = temp
    trial +=1
    temp = []
#%%
temp = []
first_frame = []
second_frame = []
trial = 0
while trial < len(frames_opto_b):
    for frame in frames_opto_b[trial]:
        if frame in frames_opto_clean_b:
            temp.append(frame)
    first_frame.append((temp[0]))
    second_frame.append((temp[1]))
    temp = []
    trial += 1
#%%
result_first_frame = np.empty((len(first_frame), 100,160))
for i in range(len(first_frame)):
    result_first_frame[i,:,:]=ds_blue_frames_norm[first_frame[i]] - bl_frames[i]
#%%
result_second_frame = np.empty((len(second_frame), 100,160))
for i in range(len(first_frame)):
    result_second_frame[i,:,:]=ds_blue_frames[second_frame[i]] - bl_frames[i]
#%%
opto_frame_is = np.zeros((len(opto_trials),2), dtype=np.int64)
opto_frames = np.zeros((len(opto_trials),50,80), dtype=np.float64)

for i in range(len(opto_trials)):
    if opto_trials[i,0] < samples_blu[-1]-10000:
        first_frame = np.argmin(abs(opto_trials[i,0] +200 - samples_blu))
        last_frame = first_frame + 2
        opto_frame_is[i,0] = first_frame
        opto_frame_is[i,1] = last_frame
#%%
for i in range(len(opto_frame_is)):
    if np.sum(bl_frame_is[i,:]) > 0:
        opto_frames[i,:,:] = np.mean(ds_blue_frames_norm[opto_frame_is[i,0]:opto_frame_is[i,1],:,:], axis=0) - bl_frames[i,:,:]
#%%
nonzero_frames = [i for i in range(len(opto_frames)) if np.sum(opto_frames[i,:,:])>0]
plt.imshow(np.mean(opto_frames[nonzero_frames], axis = 0), vmin = -3, vmax = 3, cmap = 'seismic')
#%%
bl_frame_start_i = []
for i in range(len(frame_starts)):
    bl_frame_start_i.append(np.argmin(abs(opto_trials[:,0] - samples_blu[frame_starts[i]])))
#%%
plt.imshow(np.mean( ds_blue_frames_norm[bl_frame_start_i,:,:] - bl_frames[bl_frame_start_i,:,:], axis = 0), vmin = -3, vmax = 3)
     
#%%
bl_trials = np.zeros((len(opto_trials),4), dtype=np.int32 )
for i in range(len(opto_trials)):
    if i==0:
        bl_trials[i,0] = opto_trials[i,0] - 20600
        bl_trials[i,1] = opto_trials[i,0] - 600
    if opto_trials[i,0] - opto_trials[i-1,1] > 30000:
        bl_trials[i,0] = opto_trials[i,0] - 20600
        bl_trials[i,1] = opto_trials[i,0] - 600
bl_trials = bl_trials[np.nonzero(bl_trials[:,0])]
#%
for i in range(len(bl_trials)):
    bl_trials[i,2]=np.argmin(abs(samples_blu - bl_trials[i,0]))
    bl_trials[i,3]=np.argmin(abs(samples_blu - bl_trials[i,1]))
#%
bl_trials = bl_trials[bl_trials[:,3]-bl_trials[:,2]==50]
#%
baseline_seqs = np.zeros((len(bl_trials), 125, 200))

for i in range(len(bl_trials)):
    baseline_seqs[i,:,:] = np.mean(ds_blue_frames[bl_trials[i,2]:bl_trials[i,3]], axis = 0)

grand_bl = np.mean(baseline_seqs, axis = 0)
#%%check accuracy of opto frame finder
avg_clean = np.mean(ds_blue_frames[frames_clean,:,:], axis = 0)

avg_dff = 100*(avg_clean - grand_bl)/grand_bl

plt.imshow(avg_clean, vmin = 0, vmax = 2**16)
plt.title('opto clean frames')

plt.figure()
plt.imshow(grand_bl, vmin = 0, vmax = 2**16)
plt.title('grand bl')

plt.figure()
plt.imshow(avg_clean-grand_bl, cmap='seismic', vmin = -2000, vmax = 2000)
plt.title('diff over grand bl')

plt.figure()
plt.imshow(avg_dff, cmap='seismic', vmin = -3, vmax = 3)
plt.title('dff over grand bl')
#%%
buffer = 400
bl = 400
post_opto = []
baseline = []
for i in range(len(opto_trials)-1):
    if opto_trials[i+1, 0] - opto_trials[i, 1] > 50000:
        post_opto.append(opto_trials[i,1]+buffer)
    if nidaq[3,opto_trials[i,0]-bl] < 1.5:
        baseline.append(opto_trials[i,0]-bl)

#%%
post_opto_bf = [ np.argmin(abs(f-samples_blu)) for f in post_opto if f<samples_blu[-2]]
#bl_opto_bf = [ i+200 for i in post_opto_bf if i+200<len(zs_blue_frames) ]
bl_opto_bf = [ np.argmin(abs(f-samples_blu)) for f in baseline if f<samples_blu[-2]]

z = 0.3
#%
opto_activation = np.mean(ds_blue_frames[post_opto_bf,:,:], axis = 0)
opto_bl = np.mean(ds_blue_frames[bl_opto_bf,:,:], axis = 0)
#overall_bl = np.mean(ds_blue_frames, axis = 0)
dff_opto = 100*(opto_activation - opto_bl)/opto_bl
dff_all = 100*(opto_activation - grand_bl)/grand_bl

plt.imshow(opto_activation)#, vmin = -z, vmax = z)
plt.title('opto')

plt.figure()
plt.imshow(opto_bl)#, vmin = -z, vmax = z)
plt.title('opto_bl')

plt.figure()
plt.imshow(grand_bl)
plt.title('overall bl')

plt.figure()
plt.imshow(dff_opto, cmap = 'seismic', vmin = -2, vmax = 2)
plt.title('dff optobl')

plt.figure()
plt.imshow(dff_all,  cmap = 'seismic', vmin = -1, vmax = 1)
plt.title('dff overallbl')

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