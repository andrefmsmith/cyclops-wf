# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import tifffile
import imageio
import skimage.measure
from matplotlib import animation
#%%
def load_nidaq(path, chans=10):
    '''Loads nidaq file, reshapes it and creates a dictionary with labels for each channel'''
    nidaq = np.fromfile(path, dtype=np.float64)
    #nidaq = np.reshape(nidaq, (chans, int(len(nidaq)/chans)), order='F')
    nidaq = nidaq.reshape(-1, chans).T
    
    chan_labels = {0: 'EncoderA', 1: 'EncoderB', 2: 'Photodiode', 3: 'OptoStim', 4: 'Tglobal', 5: 'UVframe', 6: 'BLUframe', 7: 'AcqImg', 8: 'RewValve', 9: 'Lick'}
    
    return nidaq, chan_labels

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
    i = 0
    while i < max(len(samples_blu), len(samples_uv))-1:
        if samples_blu[i]<samples_uv[i]:
            color_seq.append(6) #blue chan index
            if samples_uv[i]<samples_blu[i+1]:
                color_seq.append(5) #uv
            else:
                color_seq.append(6) #uv
        else:
            color_seq.append(5) #uv chan index
            if samples_blu[i]<samples_uv[i+1]:
                color_seq.append(6) #uv
            else:
                color_seq.append(5) #uv
        i+=1
    if samples_blu[-1]<samples_uv[-1]:
        color_seq.append(6) #blue chan index
        color_seq.append(5) #uv chan index
    else:
        color_seq.append(5) #uv chan index
        color_seq.append(6) #blue chan index
        
    if sum(np.diff(abs(np.diff(color_seq)))) == 0:
        print('Illumination sequence is as predicted.')
    else:
        print('Check illumination sequence.')
            
    return samples_uv, samples_blu, total_frames, total_tiffs, color_seq

def get_tif_fr_ilu(path, illum_seq):
    '''Loads the csv file that recorded frame# and tiff# for image acquired, then creates an array of tiff #, frame # and illumination color.'''
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        csvlist = list(reader)
    tiffs_frames_color = np.zeros((len(csvlist),3), dtype=np.int32)
    for tiff_i, frame_i in enumerate(csvlist):
        tiffs_frames_color[int(tiff_i), 0] = tiff_i+1
        tiffs_frames_color[int(tiff_i), 1] = frame_i[0]
        tiffs_frames_color[int(tiff_i), 2] = illum_seq[tiffs_frames_color[int(tiff_i)-1, 1]]
    return tiffs_frames_color

def load_frames(samples, tfc_array, color_seq, tiffdir, preffix='/widefield', suffix='.tif', x=500, y=800, c=6):
    '''Samples: color to be loaded.
    tfc_array: m x 3 array with tiff number, frame number and color.
    color_seq: list specifying the sequence of illumination.
    tiffdir: directory path to tiff files to be loaded.
    c: 5 (uv) or 6 (blue).
    
    Returns a t,x,y array for a single color and a list of files loaded.'''
    frame_array = np.zeros((len(samples),x,y), dtype=np.uint16)
    files_loaded = []
    
    for pair in zip(tfc_array[:,0:2]):
        tiff = pair[0][0]
        frame = pair[0][1]
    
        if color_seq[frame-1] == c:
            filename = tiffdir + preffix + str(tiff) + suffix
            frame_array[int(frame/2),:,:] = tifffile.imread(filename)
            files_loaded.append(filename)
        
    return frame_array, files_loaded

def load_and_filter(tot_frames, tfc_array, c, x=500, y=800, preffix='/widefield', suffix='.tif'):
    frame_array = np.zeros((tot_frames,x,y), dtype=np.uint16)
    files_loaded = []
    frames_loaded = []
    
    for tiff, frame in tfc_array[:,0:2]:
        if color_seq[frame-1] == c:
            filename = tiffdir + preffix + str(tiff) + suffix
            frame_array[frame-1,:,:] = tifffile.imread(filename)
            files_loaded.append(filename)
            frames_loaded.append(frame-1)
    
    return frame_array[frames_loaded], files_loaded

def smooth_pool(frame_array, x=500, y=800, k=4):
    '''Smooths a frame_array according to a box kernel of size k and reduces array size by taking mean of k-delimited box.'''
    ds_blue_frames = np.empty((frame_array.shape[0], int(x/k),int(y/k)), dtype=np.uint16)
    for t in range(frame_array.shape[0]):
        ds_blue_frames[t,:,:] = skimage.measure.block_reduce(frame_array[t,:,:], (k,k), np.mean)
    return ds_blue_frames

def animate_frameseq(frame_array, zmin, zmax, filename, fps = 25, colormap = 'seismic', savefile=True):
    fig = plt.figure()
    
    ims = []
    for i in range(frame_array.shape[0]):
        im = plt.imshow(frame_array[i,:,:], animated=True, cmap = colormap, vmin = zmin, vmax = zmax)
        ims.append([im])
        
    ani = animation.ArtistAnimation(fig, ims, interval=1000/fps, blit=True, repeat_delay=1000)
    
    if savefile == True:
        ani.save(filename+'.mp4')
    
    plt.show()

#def extract_opto_frames(nidaq_opto, wf_samples, length_opto = 9500, length_frame = 62, pre_buffer=200, post_buffer=200):
#    samples_opto = np.nonzero(np.diff(np.int32(nidaq_opto>1.5)) > 0)[0]
#    trial_start = samples_opto[1:][np.diff(samples_opto)>1000]
#    opto_trials = np.empty((len(trial_start),2), dtype='int64')
#    opto_trials[:,0] = trial_start
#    opto_trials[:,1] = trial_start + length_opto
    
#    opto_blue = []
#    for trial in range(len(opto_trials)):    
#        opto_blue.append(wf_samples[np.where(np.logical_and(wf_samples>=opto_trials[trial,0], wf_samples<=opto_trials[trial,1]))])
    
#    opto_blue_clean = []
#    opto_blue_opto = []
    
#    for trial in range(len(opto_blue)):
#        for frame in opto_blue[trial]:
#            if (nidaq_opto[frame-pre_buffer:frame+length_frame+post_buffer] < 1.5).all():
#                opto_blue_clean.append(frame)
#            else:
#                opto_blue_opto.append(frame)
                
#    frames_clean = [np.where(wf_samples == i)[0][0] for i in opto_blue_clean]
#    frames_opto = [np.where(wf_samples == i)[0][0] for i in opto_blue_opto ]
    
#    return frames_clean, frames_opto, opto_trials

    
def extract_opto_frames(nidaq_opto, wf_samples, length_opto = 9500, length_frame = 62, pre_buffer=200, post_buffer=200):
    opto_pulse_onset = [i for i in np.nonzero(np.diff(np.int32(nidaq_opto>1.5)) > 0)[0]]
    opto_trial_start = np.array(opto_pulse_onset)[np.diff([0]+opto_pulse_onset)>1000]
    opto_trial_end = opto_trial_start + length_opto
    
    frames_opto = []
    frames_opto_clean = []
    frames_opto_pulse = []
    
    for start, finish in zip(opto_trial_start, opto_trial_end):
        s = wf_samples[np.where(np.logical_and(wf_samples>=start, wf_samples<=finish))]
        if len(s)>0:
            frames_opto.append(s)
            
    for trial in range(len(frames_opto)):
        for frame in frames_opto[trial]:
            if (nidaq_opto[frame - pre_buffer:frame+length_frame+post_buffer] < 1.5).all():
                frames_opto_clean.append(np.where(wf_samples ==frame)[0][0])
            else:
                frames_opto_pulse.append(np.where(wf_samples ==frame)[0][0])
    return frames_opto_clean, frames_opto_pulse, opto_trial_start, opto_trial_end
#%%
#os.chdir('E:/WF/11.12.2019')

#tiffdir = '2019-12-11T13_52_25_WFSC01'
#nidaqfile = 'nidaq2019-12-11T13_52_14_WFSC01.bin'
#frames_csv = 'widefield2019-12-11T13_52_23_WFSC01.csv'

#nidaq, chan_labels = load_nidaq(nidaqfile)

#samples_uv, samples_blu, total_frames, total_tiffs, color_seq = illum_seq(nidaq)

#tiffs_frames_color = get_tif_fr_ilu(frames_csv, color_seq)

#blue_frames, files_loaded = load_frames(samples_blu, tiffs_frames_color, color_seq, tiffdir, c=6)

#%%
#ds_blue_frames = smooth_pool(blue_frames)
#blue_frames = None
#np.save('widefield2019-12-11T13_52_23_WFSC01', ds_blue_frames)
#%%z-score - turn into function
#zs_blue_frames = (ds_blue_frames - np.mean(ds_blue_frames, axis = 0)) / np.std(ds_blue_frames, axis = 0)
    
#animate_frameseq(zs_blue_frames[0:1000,:,:], zmin = -3, zmax = 3, filename='full_test')