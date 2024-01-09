from mvpa2.base import dataset
from mvpa2.datasets import mri
import numpy as np
from glob import glob
import re
import pandas as pd
from tqdm import tqdm
from collections.abc import Iterable
import random
import os
import sys
sys.path.append('..')
from utils import loadmat
from glm.modelspec import specify_model_funcloc, \
    specify_model_train, specify_model_test
import configs as cfg
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# =================================================================================================
# Loading functions
# =================================================================================================

def load_betas(opt, mask_templ=None, 
               bids_dir=cfg.bids_dir, fir=False, 
               max_delay=float('inf')):
        
    betas_dir = bids_dir+'derivatives/spm-preproc/derivatives/spm-stats/betas/'
        
    datamodel = get_correct_model(opt)
    data_dir = os.path.join(betas_dir, f'{opt.sub}/{opt.task}/model_{datamodel:g}/')
    if fir:
        data_dir = os.path.join(data_dir, 'FIR')
        if not os.path.isdir(data_dir):
            raise Exception('FIR not found for this model!')
    
    SPM = loadmat(os.path.join(data_dir, 'SPM.mat'))
    if fir:
        regr_names = [n[6:] for n in SPM['SPM']['xX']['name']]
    else:
        regr_names = [n[6:-6] if '*bf(1)' in n else n[6:] for n in SPM['SPM']['xX']['name']]
    
    file_names = [os.path.join(data_dir, b.fname) for b in SPM['SPM']['Vbeta']]
    
    
    exclude = ['buttonpress', 'constant', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz']
    chunk_count = {}
    for f in regr_names:
        if not f in exclude:
            chunk_count[f] = 1
    
    if mask_templ is None:
        mask_templ = os.path.join(cfg.project_dir, 'anat_roi_masks', 
                                  'wholebrain.nii')
    
    if '{:s}' in mask_templ:
        mask = mask_templ.format(opt.sub, opt.sub)
    else:
        mask = mask_templ
    
    if os.path.exists(mask):
    
        AllDS = []

        for i, f in enumerate(tqdm(file_names)):
            regr = regr_names[i]
            if not regr in exclude:
                if fir:
                    bf_n = int(re.search(r'.*bf\((\d+)\)', regr).group(1)) - 1
                    if bf_n <= max_delay:
                        # only append delays up to max_delay
                        thisDS = mri.fmri_dataset(f, chunks=chunk_count[regr], 
                                                targets=regr[:regr.find('*bf')], mask=mask)
                        thisDS.sa['delay'] = [bf_n]
                        AllDS.append(thisDS)
                else:
                    AllDS.append(mri.fmri_dataset(f, chunks=chunk_count[regr], targets=regr, mask=mask))
                chunk_count[regr] += 1

        AllDS = dataset.vstack(AllDS, a=0)

        return AllDS

    else:
        return
    
# -------------------------------------------------------------------------------------------------
    
def load_TRs(opt, TR_delay=None, mask_templ=None, bids_dir=cfg.bids_dir):
    
    if not TR_delay:
        TR_delay = [6]
    if not isinstance(TR_delay, Iterable):
        TR_delay = [TR_delay]
    
    preproc = 'smooth'
    
    # MRI data:
    data_dir = bids_dir + 'derivatives/spm-preproc/{:s}/{:s}/'.format(opt.sub, preproc)
    
    allruns = glob(data_dir+'*_task-{:s}_*_bold.nii'.format(opt.task))
    
    exclude = ['buttonpress']
    
    if mask_templ is None:
        mask_templ = os.path.join(cfg.project_dir, 'anat_roi_masks', 
                                  'wholebrain.nii')
    
    if '{:s}' in mask_templ:
        mask = mask_templ.format(opt.sub, opt.sub)
    else:
        mask = mask_templ
    
    if os.path.exists(mask):
        
        AllDS = []
        
        for i, run in enumerate(allruns):
            runno = int(run.split('run-')[1][0])
            evfile = os.path.join(bids_dir,
                                  f'{opt.sub}/func/{opt.sub}_task-{opt.task}_run-{runno:g}_events.tsv')
            
            datamodel = get_correct_model(opt)
            
            if opt.task == 'funcloc':
                events = specify_model_funcloc(evfile, datamodel)
            elif opt.task == 'train':
                events = specify_model_train(evfile, datamodel)
            elif opt.task=='test':
                behav = pd.read_csv(os.path.join(bids_dir, opt.sub, 'func',
                                                 f'{opt.sub}_task-{opt.task}_beh.tsv'), 
                                    sep='\t')
                events = specify_model_test(evfile, datamodel, behav)
            
            fullrunDS = mri.fmri_dataset(run, chunks=runno, targets='placeholder', mask=mask)

            for i, cond in enumerate(events.conditions):
                if cond not in exclude:
                    # find TR indices
                    TR_indices = []
                    TR_dels = [] # to store at which delay each sample occurred (relative to onset)
                    for j in range(len(events.onsets[i])): # onsets is a list of arrays
                        for tr in TR_delay:
                            thisonset = np.floor(events.onsets[i][j]) + tr
                            TR_indices.append(int(thisonset))
                            TR_dels.append(tr)

                    # slice dataset
                    thisDS = fullrunDS[TR_indices, :]
                    thisDS.targets = np.full(thisDS.targets.shape, cond, dtype='<U21')
                    thisDS.sa['delay'] = np.array(TR_dels)
                    thisDS.sa['TRno'] = np.array(TR_indices)
                    AllDS.append(thisDS)

        AllDS = dataset.vstack(AllDS, a=0)
        AllDS.samples = AllDS.samples.astype(float)
        return AllDS  
    
    else:
        return
    
# -------------------------------------------------------------------------------------------------
 
def load_trialbetas(opt, mask_templ=None, bids_dir=cfg.bids_dir, event=None):
    """
    """
    
    data_dir = os.path.join(bids_dir, 
                            'derivatives/spm-preproc/derivatives/spm-stats/single_trial_betas/{:s}/'.format(opt.sub))
    
    alltrials = sorted(glob(data_dir+'trial-*.nii'))
    
    if opt.model==3: # Only initial viewpoint
        alltrials = [t for t in alltrials if 'ev-2' in t]
    elif opt.model in [2, 12, 13, 15, 16, 17, 23, 24]: # Only final viewpoint
        alltrials = [t for t in alltrials if 'ev-7' in t]
        
    if event is not None: # To only load custom event
        alltrials = [t for t in alltrials if 'ev-{:g}'.format(event) in t]
        
    if mask_templ is None:
        mask_templ = os.path.join(cfg.project_dir, 'anat_roi_masks', 
                                  'wholebrain.nii')
    
    if '{:s}' in mask_templ:
        mask = mask_templ.format(opt.sub, opt.sub)
    else:
        mask = mask_templ
        
    if os.path.exists(mask):
        
        AllDS = []
        trialnos = []
        eventnos = []
    
        for tr in alltrials:
            trialno = int(re.search('trial-(.+)_', tr).group(1))
            eventno = int(re.search('ev-(.+).nii', tr).group(1))
            runno = int(np.ceil((trialno+1)/48))
            
            AllDS.append(mri.fmri_dataset(tr, chunks=runno, targets='placeholder', mask=mask))
            trialnos.append(trialno)
            eventnos.append(eventno)
        
        AllDS = dataset.vstack(AllDS, a=0)
        AllDS.sa['trialno'] = np.array(trialnos)
        AllDS.sa['eventno'] = np.array(eventnos)
        
        
        # Label
        behav = pd.read_csv(os.path.join(bids_dir,
                                         '{:s}/func/{:s}_task-{:s}_beh.tsv'.format(opt.sub, opt.sub, opt.task)), sep='\t')
        
        if opt.model==2:
            rot30_indx = behav.index[behav['FinalView']==30]
            rot90_indx = behav.index[behav['FinalView']==90]
            AllDS.sa.targets[np.isin(AllDS.sa.trialno, rot30_indx)] = 'rot30'
            AllDS.sa.targets[np.isin(AllDS.sa.trialno, rot90_indx)] = 'rot90'
            
        elif opt.model==3:
            widetrials = behav.index[behav['InitView']==1]
            narrtrials = behav.index[behav['InitView']==2]
            AllDS.sa.targets[np.isin(AllDS.sa.trialno, widetrials)] = 'wide'
            AllDS.sa.targets[np.isin(AllDS.sa.trialno, narrtrials)] = 'narrow'
        
        elif opt.model in [12, 15]:
            A30_exp = behav.index[((behav['InitView']==1)&(behav['FinalView']==30)&(behav['Consistent']==1))]
            A90_exp = behav.index[((behav['InitView']==1)&(behav['FinalView']==90)&(behav['Consistent']==1))]
            
            # Unexpected need to be swapped! (for cross-decoding)
            A30_unexp = behav.index[((behav['InitView']==2)&(behav['FinalView']==30)&(behav['Consistent']==0))]
            A90_unexp = behav.index[((behav['InitView']==2)&(behav['FinalView']==90)&(behav['Consistent']==0))]
            
            B30_exp = behav.index[((behav['InitView']==2)&(behav['FinalView']==30)&(behav['Consistent']==1))]
            B90_exp = behav.index[((behav['InitView']==2)&(behav['FinalView']==90)&(behav['Consistent']==1))]
            
            # Unexpected need to be swapped! (for cross-decoding)
            B30_unexp = behav.index[((behav['InitView']==1)&(behav['FinalView']==30)&(behav['Consistent']==0))]
            B90_unexp = behav.index[((behav['InitView']==1)&(behav['FinalView']==90)&(behav['Consistent']==0))]
            
            # Randomly divide expected in 3:
            A30_E_1 = []; A30_E_2 = []; A30_E_3 = []
            A90_E_1 = []; A90_E_2 = []; A90_E_3 = []
            B30_E_1 = []; B30_E_2 = []; B30_E_3 = []
            B90_E_1 = []; B90_E_2 = []; B90_E_3 = []
            
            for ch in np.unique(AllDS.sa.chunks): # do it separately for each run
                
                # extract trial numbers for this run
                thisruntrials = np.unique(AllDS[AllDS.sa['chunks']==ch].sa['trialno'])
                # ---------------------------------
                thisrun_A30_E = list(thisruntrials[np.isin(thisruntrials, A30_exp)])
                thisrun_A30_E = random.sample(thisrun_A30_E, len(thisrun_A30_E))
                A30_E_1.extend(thisrun_A30_E[:3])
                A30_E_2.extend(thisrun_A30_E[3:6])
                A30_E_3.extend(thisrun_A30_E[6:9])
                # ---------------------------------
                thisrun_A90_E = list(thisruntrials[np.isin(thisruntrials, A90_exp)])
                thisrun_A90_E = random.sample(thisrun_A90_E, len(thisrun_A90_E))
                A90_E_1.extend(thisrun_A90_E[:3])
                A90_E_2.extend(thisrun_A90_E[3:6])
                A90_E_3.extend(thisrun_A90_E[6:9])
                # ---------------------------------
                thisrun_B30_E = list(thisruntrials[np.isin(thisruntrials, B30_exp)])
                thisrun_B30_E = random.sample(thisrun_B30_E, len(thisrun_B30_E))
                B30_E_1.extend(thisrun_B30_E[:3])
                B30_E_2.extend(thisrun_B30_E[3:6])
                B30_E_3.extend(thisrun_B30_E[6:9])
                # ---------------------------------
                thisrun_B90_E = list(thisruntrials[np.isin(thisruntrials, B90_exp)])
                thisrun_B90_E = random.sample(thisrun_B90_E, len(thisrun_B90_E))
                B90_E_1.extend(thisrun_B90_E[:3])
                B90_E_2.extend(thisrun_B90_E[3:6])
                B90_E_3.extend(thisrun_B90_E[6:9])
             
            AllDS.sa.targets[np.isin(AllDS.sa.trialno, A30_E_1)] = 'A30_exp_1'
            AllDS.sa.targets[np.isin(AllDS.sa.trialno, A30_E_2)] = 'A30_exp_2'
            AllDS.sa.targets[np.isin(AllDS.sa.trialno, A30_E_3)] = 'A30_exp_3'
            AllDS.sa.targets[np.isin(AllDS.sa.trialno, A90_E_1)] = 'A90_exp_1'
            AllDS.sa.targets[np.isin(AllDS.sa.trialno, A90_E_2)] = 'A90_exp_2'
            AllDS.sa.targets[np.isin(AllDS.sa.trialno, A90_E_3)] = 'A90_exp_3'
            
            AllDS.sa.targets[np.isin(AllDS.sa.trialno, B30_E_1)] = 'B30_exp_1'
            AllDS.sa.targets[np.isin(AllDS.sa.trialno, B30_E_2)] = 'B30_exp_2'
            AllDS.sa.targets[np.isin(AllDS.sa.trialno, B30_E_3)] = 'B30_exp_3'
            AllDS.sa.targets[np.isin(AllDS.sa.trialno, B90_E_1)] = 'B90_exp_1'
            AllDS.sa.targets[np.isin(AllDS.sa.trialno, B90_E_2)] = 'B90_exp_2'
            AllDS.sa.targets[np.isin(AllDS.sa.trialno, B90_E_3)] = 'B90_exp_3'
            
            AllDS.sa.targets[np.isin(AllDS.sa.trialno, A30_unexp)] = 'A30_unexp'
            AllDS.sa.targets[np.isin(AllDS.sa.trialno, A90_unexp)] = 'A90_unexp'
            AllDS.sa.targets[np.isin(AllDS.sa.trialno, B30_unexp)] = 'B30_unexp'
            AllDS.sa.targets[np.isin(AllDS.sa.trialno, B90_unexp)] = 'B90_unexp'
            
        elif opt.model==13:
            E30_indx = behav.index[((behav['FinalView']==30)&(behav['Consistent']==1))]
            E90_indx = behav.index[((behav['FinalView']==90)&(behav['Consistent']==1))]
            
            U30_indx = behav.index[((behav['FinalView']==30)&(behav['Consistent']==0))]
            U90_indx = behav.index[((behav['FinalView']==90)&(behav['Consistent']==0))]
            
            # Randomly divide expected in 3:
            E30_1 = []; E30_2 = []; E30_3 = []
            E90_1 = []; E90_2 = []; E90_3 = []
            
            for ch in np.unique(AllDS.sa.chunks):
                thisruntrials = np.unique(AllDS[AllDS.sa['chunks']==ch].sa['trialno'])
                # ---------------------------------
                thisrun_E30 = list(thisruntrials[np.isin(thisruntrials, E30_indx)])
                thisrun_E30 = random.sample(thisrun_E30, len(thisrun_E30))
                E30_1.extend(thisrun_E30[:6])
                E30_2.extend(thisrun_E30[6:12])
                E30_3.extend(thisrun_E30[12:18])
                # ---------------------------------
                thisrun_E90 = list(thisruntrials[np.isin(thisruntrials, E90_indx)])
                thisrun_E90 = random.sample(thisrun_E90, len(thisrun_E90))
                E90_1.extend(thisrun_E90[:6])
                E90_2.extend(thisrun_E90[6:12])
                E90_3.extend(thisrun_E90[12:18])
                
            AllDS.sa.targets[np.isin(AllDS.sa.trialno, E30_1)] = 'rot30_exp_1'
            AllDS.sa.targets[np.isin(AllDS.sa.trialno, E30_2)] = 'rot30_exp_2'
            AllDS.sa.targets[np.isin(AllDS.sa.trialno, E30_3)] = 'rot30_exp_3'
            AllDS.sa.targets[np.isin(AllDS.sa.trialno, E90_1)] = 'rot90_exp_1'
            AllDS.sa.targets[np.isin(AllDS.sa.trialno, E90_2)] = 'rot90_exp_2'
            AllDS.sa.targets[np.isin(AllDS.sa.trialno, E90_3)] = 'rot90_exp_3'
            
            AllDS.sa.targets[np.isin(AllDS.sa.trialno, U30_indx)] = 'rot30_unexp'
            AllDS.sa.targets[np.isin(AllDS.sa.trialno, U90_indx)] = 'rot90_unexp'
        
        elif opt.model in [16, 17]:
            
            A30mask = ((behav['InitView']==1)&(behav['FinalView']==30))
            A90mask = ((behav['InitView']==1)&(behav['FinalView']==90))
            B30mask = ((behav['InitView']==2)&(behav['FinalView']==30))
            B90mask = ((behav['InitView']==2)&(behav['FinalView']==90))
            
            A30_indx = behav.index[((behav['Consistent']==1) & A30mask) | ((behav['Consistent']==0) & B30mask)]
            A90_indx = behav.index[((behav['Consistent']==1) & A90mask) | ((behav['Consistent']==0) & B90mask)]
            B30_indx = behav.index[((behav['Consistent']==1) & B30mask) | ((behav['Consistent']==0) & A30mask)]
            B90_indx = behav.index[((behav['Consistent']==1) & B90mask) | ((behav['Consistent']==0) & A90mask)]
            
            AllDS.sa.targets[np.isin(AllDS.sa.trialno, A30_indx)] = 'A30'
            AllDS.sa.targets[np.isin(AllDS.sa.trialno, A90_indx)] = 'A90'
            AllDS.sa.targets[np.isin(AllDS.sa.trialno, B30_indx)] = 'B30'
            AllDS.sa.targets[np.isin(AllDS.sa.trialno, B90_indx)] = 'B90'
            
        elif opt.model=='all':
            # Just get all trials and events
            pass
            
            
        return AllDS
            
    else:
        return

# -------------------------------------------------------------------------------------------------

def get_correct_model(opt):
    """
    Some models are just different labelings
    of data estimated from other models.
    """
    if opt.task=='test' and opt.model==15:
        return 12
    elif opt.task=='test' and opt.model==17:
        return 16
    elif opt.task=='test' and opt.model==21:
        return 20
    elif opt.task=='test' and opt.model==24:
        return 23
    elif opt.task=='test' and opt.model==29:
        return 28
    elif opt.task=='train' and opt.model==5:
        return 4
    else:
        return opt.model
    

# -------------------------------------------------------------------------------------------------
    
if __name__=="__main__":
    print('Base directory:', cfg.base_dir)