import pandas as pd
import numpy as np
import os
from glob import glob
import sys
sys.path.append('/project/3018040.05/rotscenetask_fmri/analysis/')
from mvpa.loading import load_betas, get_correct_model
from mvpa.mvpa_utils import correct_labels
from utils import Options, loadmat
from configs import project_dir, bids_dir
from readresults.readres_mvpa import parse_roi_info
import argparse


def get_contrast_files(subjlist, task, model, contrast, base_dir=bids_dir):
    contr_dir = os.path.join(bids_dir, 'derivatives', 'spm-preproc',
                             'derivatives', 'spm-stats', 'contrasts')
    contrastfiles = []
    for s in subjlist:
        contrastfiles.append(os.path.join(contr_dir, s, task, f'model_{model}',
                                          f'con_{contrast:04d}.nii'))
    return contrastfiles


def load_univar_by_voxelno(sub, roi_templ, task, model, voxelnos):
    """
    Returns a dataframe containing the average activation by number of voxels
    """
    alldata = []
    for vn in voxelnos:
        for h in ['L', 'R']:
            alldata.append(load_univariate(sub, roi_templ.format(h, vn),
                                           task, model))
    alldata = parse_roi_info(pd.concat(alldata))
    return alldata            
    

def load_univariate(sub, roi, task, model):
    """
    Returns a dataframe containing
    the average activation (beta) in the given ROI.
    """
    
    if 'contr' in roi: # functional contrast
        roi_basedir = os.path.join(bids_dir, 'derivatives', 'spm-preproc', 
                                   'derivatives', 'roi-masks')
        mask_templ = os.path.join(roi_basedir, '{:s}/{:s}_' + roi + '.nii')
    else: # only anatomical map
        roi_basedir = os.path.join(project_dir, 'anat_roi_masks')
        mask_templ = os.path.join(roi_basedir, roi + '.nii')
    
    opt = Options(
        sub=sub,
        task=task,
        model=model
    )
    
    DS = load_betas(opt, mask_templ=mask_templ)
    
    if DS is not None:
        
        DS = correct_labels(DS, opt)
        nanmask = np.all(np.isfinite(DS.samples), axis=0)
        DS = DS[:, nanmask]
        
        univar_df = pd.DataFrame(
            {'mean_beta': list(np.mean(DS.samples, axis=1)),
             'condition': list(DS.sa.targets),
             'subject': [sub]*len(DS),
             'roi': [roi]*len(DS)}
        )
        
        return univar_df
    
    else:
        
        return None
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub', required=True, type=str, help="Subject")
    args = parser.parse_args()
    
    roi = 'ba-17-18_{:s}_contr-objscrvsbas_top-{:g}_nothresh'
    voxelnos = np.arange(100, 6100, 100)
    univar_df = load_univar_by_voxelno(args.sub, roi, 'test', 5, voxelnos)
    univar_df.to_csv(f'/project/3018040.05/Univar_results/test_m5/{args.sub}.csv', index=False)