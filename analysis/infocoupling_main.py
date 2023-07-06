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
        
        DS = load_TRs(opt.sub, opt.task, 
                      opt.model, TR_delay=delays, \
                          mask_templ=mask_templ)
        
        if DS is not None:
            
            DS = correct_labels(DS, opt)
            DS = DS.remove_nonfinite_features()
            res = decode_CV(DS, opt)
            
    elif approach=='traintest':
        
        train_opt, test_opt = split_options(opt)