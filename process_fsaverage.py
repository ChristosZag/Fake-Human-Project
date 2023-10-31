#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 12:04:03 2023

@author: christoszangos
"""

import nibabel as nib
import os
import h5py
import json
import pickle
import numpy as np
import scipy.io as spio
from tqdm import tqdm
from scipy.stats import zscore


FMRI_DIR='/Users/christoszangos/Desktop/neurips/ICLR/not_zscored/RAW_SUBJ02'  
ROI_DIR='/Users/christoszangos/Desktop/neurips/ICLR/ROI' 
OUT_DIR='/Users/christoszangos/Desktop/Subject02_before_zscore' 
zscored_dir='/Users/christoszangos/Desktop/Subject02_zscored'
ZSCORED_DIR = zscored_dir+'/nsdgeneral_zscored' # Directory containing the processed fMRIS (after z-score) will be stored
SESS_NUM=37




def extract_voxels_fsaverage(fmri_dir,roi_dir, out_dir, flatten=True):
    
    """
    Extracts voxel information based on ROIs (regions of interest) from the fMRI data of fsaverage common space.

    Parameters:
    - fmri_dir: Directory containing fMRI data downloaded from the NSD database.
                https://natural-scenes-dataset.s3.amazonaws.com/index.html#nsddata_betas/ppdata/subj01/fsaverage/betas_fithrf_GLMdenoise_RR/
                
    - roi_dir: Directory containing ROI masks for left and right hemispheres downloaded from the NSD database (i.e. rh.nsdgeneral.mgz and lh.nsdgeneral.mgz).
    - out_dir: Directory to save the processed output (before z-score).
    - flatten: Boolean flag (default True) to flatten the ROI masks.

    Returns:
    - No explicit return, saves processed data in HDF5 format in the out_dir.

    Process:
    - Loads ROI masks for left and right hemispheres.
    - Identifies available regions in each ROI.
    - Transposes and extracts voxel data based on ROIs from the fMRI data.
    - Saves the processed data in HDF5 format in the out_dir.
    """
    
    rois=os.listdir(roi_dir)
    for roi_file in rois:
        roi_name=os.path.basename(roi_file)[:-4] # Extract the ROI name
        hem=roi_name[:2] # Identify the hemisphere (lh or rh)
        roi_file=os.path.join(roi_dir,roi_file) # Path to ROI file
        
        # Load and process ROI mask based on hemisphere
        if hem=='lh':
            _mask_lh=nib.load(roi_file).get_fdata() # Load left hemisphere ROI
            
            available_region = [int(r) for r in set(_mask_lh.flatten())] # Identify available regions
            print(f'Extracting ROI based on {roi_name},',
                  f'available_regions: {available_region}')
            
        else: 
            _mask_rh=nib.load(roi_file).get_fdata() # Load right hemisphere ROI
            available_region = [int(r) for r in set(_mask_rh.flatten())] # Identify available regions
            print(f'Extracting ROI based on {roi_name},',
                  f'available_regions: {available_region}')
            
    # Process fMRI data    
    mask_rh=(_mask_rh != 0) # Generate mask for right hemisphere
    mask_lh=(_mask_lh != 0) # Generate mask for left hemisphere
    print(f'\nTotal ROI voxel count for lh: {np.count_nonzero(mask_lh)}\n', flush=True)
    print(f'\nTotal ROI voxel count for rh: {np.count_nonzero(mask_rh)}\n', flush=True)
    
    # Retrieve a list of fMRI files
    fmri_list=os.listdir(fmri_dir)


    sess_list=[]
    
    # Prepare sessions based on file naming convention
    for sess in range(1,int(len(fmri_list)/2)+1):
        
        sess_list.append([file for file in fmri_list if os.path.basename(file)[-6:-4]==f'{sess:02d}'])
        
        
    # Process fMRI data based on sessions    
    for i in tqdm(sess_list):
        if len(i)!=2:
            print(f'Run number {sess_list.index(i)+1} does not include both hemispheres!')
            continue
        for fmri_file in i:
            
            
            hem=os.path.basename(fmri_file)[:2]
            fmri_file=os.path.join(fmri_dir,fmri_file)
            fmri = nib.load(fmri_file).get_fdata()
            fmri=np.transpose(fmri, (3, 0, 1, 2))
            
            
            # Extract voxel data for each hemisphere
            
            if hem=="lh":
                fmri = [fmri[trial][mask_lh] for trial in range(len(fmri))]
                fmri_lh = np.stack(fmri)
                type(fmri_lh)
            
            else:
                
                fmri = [fmri[trial][mask_rh] for trial in range(len(fmri))]
                fmri_rh = np.stack(fmri)
                type(fmri_rh)
                
        # Combine left and right hemisphere data
        fmri=np.concatenate((fmri_lh, fmri_rh), axis=1)
        fmri_rh=0
        fmri_lh=0
            
        out_f = os.path.join(out_dir, os.path.basename(fmri_file)[3:-4]+'_fsaverage.hdf5')
        
        # Save processed data in HDF5 format
        with h5py.File(out_f, 'w') as f:
            dset = f.create_dataset('betas', data=fmri)
            
            
            
            
           
            

if __name__=='__main__':
    
    extract_voxels_fsaverage(FMRI_DIR,ROI_DIR, OUT_DIR)
    
    n = 3 # repetition
    
    #Z-SCORE PROCESS
    for sess in tqdm(range(SESS_NUM)):
        in_file = os.path.join(OUT_DIR, f'betas_session{sess+1:02}_fsaverage.hdf5')
        out_file = os.path.join(ZSCORED_DIR, f'betas_session{sess+1:02}.hdf5')
        with h5py.File(in_file, 'r') as f:
            fmri = f['betas'][()]
        fmri = zscore(fmri, 0)
        with h5py.File(out_file, 'w') as f:
            dset = f.create_dataset('betas', data=fmri)
         
        
        
    '''
    # Here you can work with the dataset_data as a regular numpy array. Run seperately!
    
    test_file='/Users/christoszangos/Desktop/Subject02_zscored/nsdgeneral_zscored/betas_session33.hdf5'
    with h5py.File(test_file, 'r') as hdf5_file:
        # List the datasets in the HDF5 file.
        print("Datasets in the HDF5 file:", list(hdf5_file.keys()))
    
        # Access a specific dataset.
        dataset = hdf5_file["betas"]
    
        # Get the data as a numpy array.
        dataset_data = dataset[()]
    
        # Optionally, you can access dataset attributes if they exist.
        dataset_attributes = dict(dataset.attrs)
    
    
    print("Shape of the dataset array:", dataset_data.shape)
    print("Data type of the dataset array:", dataset_data.dtype)
    '''
        