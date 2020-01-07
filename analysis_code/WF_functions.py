# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import tifffile
import imageio
import skimage.measure
from matplotlib import animation

os.chdir('E:/WF/11.12.2019')
#%%
tiffdir = '2019-12-11T13_52_25_WFSC01'
nidaqfile = 'nidaq2019-12-11T13_52_14_WFSC01.bin'
frames_csv = 'widefield2019-12-11T13_52_23_WFSC01.csv'
#%%
def load_nidaq(path, chans=10):
    '''Loads nidaq file, reshapes it and creates a dictionary with labels for each channel'''
    nidaq = np.fromfile(path, dtype=np.float64)
    #nidaq = np.reshape(nidaq, (chans, int(len(nidaq)/chans)), order='F')
    nidaq = nidaq.reshape(-1, chans).T
    
    chan_labels = {0: 'EncoderA', 1: 'EncoderB', 2: 'Photodiode', 3: 'OptoStim', 4: 'Tglobal', 5: 'UVframe', 6: 'BLUframe', 7: 'AcqImg', 8: 'RewValve', 9: 'Lick'}
    
    return nidaq, chan_labels

def illum_seq(nidaq, u=5, b=6):
    '''1. Obtains UV and Blue illumination onsets from nidaq.
    2. Calculates total_frames and total_tiffs captured.
    3. Establishes the illumination sequence sent during the experiment.
    4 Returns this sequence, UV and blue sample onsets, total_frames and total_tiffs captured.'''
    
    uv = np.diff(np.int32(nidaq[u,:]>1.5)) > 0
    blu = np.diff(np.int32(nidaq[b,:]>1.5)) > 0
    samples_uv = np.nonzero(uv)[0]
    samples_blu = np.nonzero(blu)[0]
    total_frames = len(samples_uv) + len(samples_blu)
    total_tiffs = len(os.listdir(tiffdir))
    
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
    path = frames_csv
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        csvlist = list(reader)
    tiffs_frames_color = np.zeros((len(csvlist),3), dtype=np.int32)
    for tiff_i, frame_i in enumerate(csvlist):
        tiffs_frames_color[int(tiff_i), 0] = tiff_i+1
        tiffs_frames_color[int(tiff_i), 1] = frame_i[0]
        tiffs_frames_color[int(tiff_i), 2] = illum_seq[tiffs_frames_color[int(tiff_i)-1, 1]]
    return tiffs_frames_color

def smooth_pool(frame_array, x=500, y=800, k=4):
    '''Smooths a frame_array according to a box kernel of size k and reduces array size by taking mean of k-delimited box.'''
    ds_blue_frames = np.empty((frame_array.shape[0], int(x/k),int(y/k)), dtype=np.uint16)
    for t in range(frame_array.shape[0]):
        ds_blue_frames[t,:,:] = skimage.measure.block_reduce(frame_array[t,:,:], (k,k), np.mean)
    return ds_blue_frames
#%%
nidaq, chan_labels = load_nidaq(nidaqfile)
samples_uv, samples_blu, total_frames, total_tiffs, color_seq = illum_seq(nidaq)
#color_seq=illum_seq(samples_blu, samples_uv)
tiffs_frames_color = get_tif_fr_ilu(frames_csv, color_seq)
#%%
blue_frames = np.zeros((len(samples_blu),500,800), dtype=np.uint16)

#%%Load frames into reconstructed full array - turn into function
#def load_imaging_frames(tiffs_frames_color, color_seq, tiffdir, preffix = '/widefield', suffix = '.tif')

preffix = '/widefield'
suffix = '.tif'
files_loaded = []

start = time.time()
for pair in zip(tiffs_frames_color[:,0:2]):
    tiff = pair[0][0]
    frame = pair[0][1]
    
    if color_seq[frame-1] == 6:
        filename = tiffdir + preffix + str(tiff) + suffix
        blue_frames[int(frame/2),:,:] = tifffile.imread( filename )
        files_loaded.append(filename)
elapsed_time_fl = (time.time()-start)
print(elapsed_time_fl/60)
#%%
ds_blue_frames = smooth_pool(blue_frames)
blue_frames = None
np.save('widefield2019-12-11T13_52_23_WFSC01', ds_blue_frames)
#%%z-score - turn into function
zs_blue_frames = (ds_blue_frames - np.mean(ds_blue_frames, axis = 0)) / np.std(ds_blue_frames, axis = 0)

#%%

def animate_frameseq(frame_array, zmin, zmax, filename, fps = 25, colormap = 'seismic'):
    fig = plt.figure()
    
    ims = []
    for i in range(frame_array.shape[0]):
        im = plt.imshow(frame_array[i,:,:], animated=True, cmap = colormap, vmin = zmin, vmax = zmax)
        ims.append([im])
        
    ani = animation.ArtistAnimation(fig, ims, interval=1000/fps, blit=True, repeat_delay=1000)
    ani.save(filename+'.mp4')
    
    plt.show()
    
#%%
    
animate_frameseq(zs_blue_frames[0:1000,:,:], zmin = -3, zmax = 3, filename='full_test')