import pandas as pd
import numpy as np
import os
import sys
sys.path.append('/project/3018040.05/rotscenetask_fmri/analysis/')
from mvpa.loading import load_betas, get_correct_model
from mvpa.mvpa_utils import correct_labels
from utils import Options, loadmat
from configs import project_dir, bids_dir
from readresults.readres_mvpa import parse_roi_info
import argparse

def load_labeled_contrasts(subj_list):
    """
    'opt' should contain sub, task, model
    """
    contr_dir = bids_dir+'derivatives/spm-preproc/derivatives/spm-stats/contrasts/'
    datamodel = get_correct_model(opt)
    data_dir = os.path.join(contr_dir, f'{opt.sub}/{opt.task}/model_{datamodel:g}/')
    
    SPM = loadmat(os.path.join(data_dir, 'SPM.mat'))
    regr_names = [n[6:-6] if '*bf(1)' in n else n[6:] for n in SPM['SPM']['xX']['name']]
    file_names = [os.path.join(data_dir, b.fname) for b in SPM['SPM']['Vbeta']]
    
    exclude = ['buttonpress', 'constant', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz']
    
    file_names = [f for f, r in zip(file_names, regr_names) if r not in exclude]
    regr_names = [r for r in regr_names if r not in exclude]

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