from nipype import Node, Workflow, IdentityInterface, Function
from nipype.algorithms import modelgen
from nipype.interfaces import spm
from nipype.interfaces.spm.model import Level1Design, EstimateModel, EstimateContrast
from nipype.interfaces.base import Bunch
import nipype.interfaces.io as nio

def decode_timecourses(opt, approach, func_runs, motpar):
    """
    - opt: should contain sub, roi, task, model
    """
    import os
    import numpy as np
    import pandas as pd
    import sys
    sys.path.append('/project/3018040.05/rotscenetask_fmri/analysis/')
    from mvpa.loading import load_TRs
    from mvpa.decoding import decode_CV, decode_traintest
    from mvpa.mvpa_utils import correct_labels
    from configs import project_dir, bids_dir
    from utils import split_options
    
    if 'contr' in opt.roi: # functional contrast
        roi_basedir = os.path.join(bids_dir, 'derivatives', 'spm-preproc', 
                                   'derivatives', 'roi-masks')
        mask_templ = os.path.join(roi_basedir, '{:s}/{:s}_' + opt.roi + '.nii')
    else: # only anatomical map
        roi_basedir = os.path.join(project_dir, 'anat_roi_masks')
        mask_templ = os.path.join(roi_basedir, opt.roi + '.nii')
        
    delays = range(19)
    
    if approach=='CV':
        
        DS = load_TRs(opt, TR_delay=delays, \
                      mask_templ=mask_templ)
        
        if DS is not None:
            
            DS = correct_labels(DS, opt)
            DS = DS.remove_nonfinite_features()
            
            allres = []
            
            for d in delays:
                thisDS = DS[DS.sa.delay==d]
                allres.append(decode_CV(thisDS, opt))
            
            allres = pd.concat(allres)
            
            return allres #, approach, func_runs, motpar
        
        else:
            
            return np.nan
            
    elif approach=='traintest':
        
        train_opt, test_opt = split_options(opt)
        
        # only option for now, maybe implement more later
        assert train_opt.task=='train' and test_opt.task=='test'
        
        # The TR delays here correspond to the start and end of the 
        # miniblock, shifted by 6s. They're used all together.
        trainDS = load_TRs(train_opt, TR_delay=range(6, 20), \
                            mask_templ=mask_templ)
        
        if trainDS is not None:
            
            trainDS = correct_labels(trainDS, train_opt)
            
            testDS = load_TRs(test_opt, TR_delay=delays,
                              mask_templ=mask_templ)
            testDS = correct_labels(testDS, test_opt)
            
            nanmask = np.logical_and(np.all(np.isfinite(trainDS.samples), axis=0), \
                np.all(np.isfinite(testDS.samples), axis=0))
            trainDS = trainDS[:, nanmask]
            testDS = testDS[:, nanmask]
            
            allres = []
            
            for d in delays:
                thistestDS = testDS[testDS.sa.delay==d]
                thisres = decode_traintest(trainDS, thistestDS, \
                    train_opt, test_opt)
            
            allres = pd.concat(allres)
            
            return allres
        
        else:
            
            return np.nan
        
# ---------------------------------------------------------------------------------

def extract_timecourse(res, approach, func_runs=None, motpar=None):
    
        