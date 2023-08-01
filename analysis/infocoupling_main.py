from nipype import Node, Workflow, IdentityInterface, Function
from nipype.algorithms import modelgen
from nipype.interfaces import spm
from nipype.interfaces.spm.model import Level1Design, EstimateModel, EstimateContrast
from nipype.interfaces.base import Bunch
import nipype.interfaces.io as nio
import ipdb

def decode_timecourses(sub, roi, task, model, approach, func_runs=None, motpar=None):
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
    from utils import Options, split_options
    
    opt = Options(
            sub=sub,
            roi=roi,
            task=task,
            model=model
            )
    
    if 'contr' in opt.roi: # functional contrast
        roi_basedir = os.path.join(bids_dir, 'derivatives', 'spm-preproc', 
                                   'derivatives', 'roi-masks')
        mask_templ = os.path.join(roi_basedir, '{:s}/{:s}_' + opt.roi + '.nii')
    else: # only anatomical map
        roi_basedir = os.path.join(project_dir, 'anat_roi_masks')
        mask_templ = os.path.join(roi_basedir, opt.roi + '.nii')
        
    delays = range(9)
    
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
            
        else:
            
            allres = None
            
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
                allres.append(thisres)
            
            allres = pd.concat(allres)
            
            if 'expected' in allres.columns:
                assert 'split' in allres.columns
                allres = allres.sort_values(by=['chunk', 'TRno', 'expected', 'split'])
            else:
                allres = allres.sort_values(by=['chunk', 'TRno'])
        
        else:
            
            allres = None
        
    if allres is not None:
        allres['subject'] = sub
        allres['roi'] = roi
        allres['approach'] = approach
        if isinstance(task, tuple):
            allres['traintask'] = task[0]
            allres['testtask'] = task[1]
            allres['trainmodel'] = model[0]
            allres['testmodel'] = model[1]
        else:
            allres['traintask'] = task
            allres['testtask'] = task
            allres['trainmodel'] = model
            allres['testmodel'] = model
        
        return allres
    
    else:
        return np.nan
            
# ---------------------------------------------------------------------------------

def save_timecourse(tc, sub, roi):
    import os
    
    datadir = '/project/3018040.05/rotscenetask_fmri/analysis/infocoupling/'
    filepath = os.path.join(datadir, f'TR_timecourses/{sub}_{roi}.csv')
    tc.to_csv(filepath, index=False)
    
    return
        
# ---------------------------------------------------------------------------------

def add_motion_regressors_infocoupl(subj_info, task, 
                                    func_runs, motpar,
                                    use_motion_reg=True):
    """
    - subj_info: output of ModelSpecify
    - use_motion_reg: bool,
    - motpar: list of .txt files
    """
    import sys
    sys.path.append('/project/3018040.05/rotscenetask_fmri/analysis/')
    from glm.motionparameters import read_motion_par
    
    if use_motion_reg:
        for run in range(len(subj_info)):
            subj_info[run].regressor_names += ['tx', 'ty', 'tz',
                                               'rx', 'ry', 'rz']
            subj_info[run].regressors += read_motion_par(motpar, task, run) # list of 6 columns
            
    return subj_info, func_runs

# ---------------------------------------------------------------------------------

def main():
    
    subjlist = ['sub-{:03d}'.format(i) for i in range(1, 36)]
    roilist = ['ba-17-18_contr-objscrvsbas_top-500']
    
    # ------------------------------------------------------
    # Utilities
    # ------------------------------------------------------
    
    # Identity interface
    subjinfo = Node(IdentityInterface(fields=['sub', 'roi']), name='subjinfo')
    subjinfo.iterables = [('sub', subjlist), ('roi', roilist)]
    
    # Datasink
    datasink = Node(nio.DataSink(parameterization=True), name='datasink')
    datasink.inputs.base_directory = '/project/3018040.05/InfoCoupling'
    subs = [('_sub_', '_'), ('_roi_', '')]
    datasink.inputs.substitutions = subs
    
    # ------------------------------------------------------
    # Custom nodes
    # ------------------------------------------------------
    
    # sub, roi, func_runs, motpar
    all_options = Node(Function(input_names = ['sub', 'roi', 'task', 'preproc', 'base_directory'],
                                output_names = ['sub', 'roi', 'func_runs', 'motpar'],
                                function = alloptions), name = 'all_options')
    all_options.inputs.task = 'test'
    all_options.inputs.preproc = 'smooth'
    all_options.inputs.base_directory = '/project/3018040.05/bids'
    
    decode_timecourses = Node(Function(input_names = ['sub', 'roi', 'approach',
                                                      'task', 'func_runs', 'motpar'],
                                   output_names = ['res', 'approach', 'func_runs', 'motpar'],
                                   function = decode_timecourses), name = 'decode_timecourses')
    decode_timecourses.inputs.approach = 'traintest'
    decode_timecourses.inputs.task = ('train', 'test')

# ---------------------------------------------------------------------------------

#def extract_timecourse(res, approach, func_runs=None, motpar=None):
    
if __name__=="__main__":
    
    from utils import Options
    
    opt = Options(
        sub='sub-001',
        task=('train', 'test'),
        model=(5, 15),
        roi='ba-17-18_L_contr-objscrvsbas_top-1000'
    ) 
    
    allres = decode_timecourses(opt, 'traintest')      