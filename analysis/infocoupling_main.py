from nipype import Node, Workflow, IdentityInterface, Function
from nipype.algorithms import modelgen
from nipype.interfaces import spm
from nipype.interfaces.spm.model import Level1Design, EstimateModel, EstimateContrast
from nipype.interfaces.base import Bunch
import nipype.interfaces.io as nio
from nipype.interfaces.matlab import MatlabCommand
from configs import project_dir, spm_dir, matlab_cmd
MatlabCommand.set_default_matlab_cmd(matlab_cmd)
MatlabCommand.set_default_paths(spm_dir)
spm.SPMCommand.set_mlab_paths(matlab_cmd=matlab_cmd)
from general_utils import Options
import os

# ---------------------------------------------------------------------------------

def get_func_files(sub, roi, task, model, approach, preproc='smooth'):
    """
    
    """
    import os
    from glob import glob
    from nipype.utils.misc import human_order_sorted
    import sys
    sys.path.append('/project/3018040.05/rotscenetask_fmri/analysis/')
    from configs import bids_dir
    
    if approach == 'CV':
        univar_task = task
    elif approach == 'traintest':
        univar_task = task[1]
    
    # Functional runs for the given task
    func_runs = glob(os.path.join(bids_dir, 'derivatives',
                                  'spm-preproc', sub, preproc,
                                  f'*_task-{univar_task}_*_bold.nii'))
    func_runs = human_order_sorted(func_runs)
    
    motpar = glob(os.path.join(bids_dir, 'derivatives',
                               'spm-preproc', sub, 'realign_unwarp',
                               'rp_*.txt'))
    motpar = human_order_sorted(motpar)
    print('Finished getting files!')
    return sub, roi, task, model, approach, func_runs, motpar, univar_task
    
# ---------------------------------------------------------------------------------    

def decode_timecourses(sub, roi, task, model, 
                       approach, dataformat='TRs',
                       func_runs=None, motpar=None):
    """
    """
    import os
    import numpy as np
    import pandas as pd
    import sys
    sys.path.append('/project/3018040.05/rotscenetask_fmri/analysis/')
    from mvpa.loading import load_TRs, load_betas
    from mvpa.decoding import decode_CV, decode_traintest
    from mvpa.mvpa_utils import correct_labels
    from configs import project_dir, bids_dir
    from general_utils import Options, split_options
    
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
        
    max_delay = 9
    
    if dataformat == 'TRs':
        loadfun = lambda opt: load_TRs(opt, TR_delay=range(max_delay+1),
            mask_templ=mask_templ)
    elif dataformat == 'FIR':
        loadfun = lambda opt: load_betas(opt, mask_templ=mask_templ, 
                                         fir=True, max_delay=max_delay)
    
    if approach=='CV':
        
        DS = loadfun(opt)
        
        if DS is not None:
            
            DS = correct_labels(DS, opt)
            DS = DS.remove_nonfinite_features()
            
            allres = []
            
            for d in range(max_delay):
                thisDS = DS[DS.sa.delay==d]
                allres.append(decode_CV(thisDS, opt))
            
            allres = pd.concat(allres)
            
        else:
            
            allres = None
            
    elif approach=='traintest':
        
        train_opt, test_opt = split_options(opt)
        
        # only option for now, maybe implement more later
        assert train_opt.task=='train' and test_opt.task=='test'
        
        if dataformat == 'TRs':
            # The TR delays here correspond to the start and end of the 
            # miniblock, shifted by 6s. They're used all together.
            trainDS = load_TRs(train_opt, TR_delay=range(6, 20), \
                                mask_templ=mask_templ)
        elif dataformat == 'FIR':
            # Just normal betas for training
            trainDS = load_betas(train_opt, mask_templ=mask_templ,
                                 fir=False)
        
        if trainDS is not None:
            
            trainDS = correct_labels(trainDS, train_opt)
            
            testDS = loadfun(test_opt)
            testDS = correct_labels(testDS, test_opt)
            
            nanmask = np.logical_and(np.all(np.isfinite(trainDS.samples), axis=0), \
                np.all(np.isfinite(testDS.samples), axis=0))
            trainDS = trainDS[:, nanmask]
            testDS = testDS[:, nanmask]
            
            allres = []
            
            for d in range(max_delay+1):
                thistestDS = testDS[testDS.sa.delay==d]
                thisres = decode_traintest(trainDS, thistestDS, \
                    train_opt, test_opt)
                allres.append(thisres)
            
            allres = pd.concat(allres)
            
            if dataformat == 'TRs':
                if 'expected' in allres.columns:
                    assert 'split' in allres.columns
                    allres = allres.sort_values(by=['runno', 'TRno', 'expected', 'split'])
                else:
                    allres = allres.sort_values(by=['runno', 'TRno'])
        
        else:
            
            allres = None
    
    #pdb.set_trace()    
    if allres is not None:
        allres['subject'] = opt.sub
        allres['roi'] = opt.roi
        allres['approach'] = approach
        if approach == 'traintest':
            allres['traintask'] = train_opt.task
            allres['testtask'] = test_opt.task
            allres['trainmodel'] = train_opt.model
            allres['testmodel'] = test_opt.model
        elif approach == 'CV':
            allres['traintask'] = opt.task
            allres['testtask'] = opt.task
            allres['trainmodel'] = opt.model
            allres['testmodel'] = opt.model
        
        return allres, func_runs, motpar
    
    else:
        return np.nan, func_runs, motpar
            
# ---------------------------------------------------------------------------------

def save_timecourse(tc, sub, roi):
    import os
    
    datadir = '/project/3018040.05/rotscenetask_fmri/analysis/infocoupling/'
    filepath = os.path.join(datadir, f'TR_timecourses/{sub}_{roi}.csv')
    tc.to_csv(filepath, index=False)
    
    return
        
# ---------------------------------------------------------------------------------

def add_evidence_regressor(tc, func_runs, motpar):
    import numpy as np
    from nipype.interfaces.base import Bunch
    
    subj_info = []
    
    for run in tc.runno.unique():
        
        regressor_names = []
        regressors = []
        thisrun = tc[tc['runno']==run]
        for i, exp in [(True, 'expected'), (False, 'unexpected')]:
            regressor_names.append(exp)
            thiscond = thisrun[thisrun['expected']==i]
            allTRs = np.zeros((404, 1))
            for _, r in thiscond.iterrows():
                allTRs[r.TRno] = r.distance
            regressors.append(allTRs.tolist())
    
        subj_info.append(Bunch(conditions=['dummy'], onsets=[[0.0]],
                               durations=[[0.0]], 
                               regressor_names=regressor_names,
                               regressors=regressors))
    
    return subj_info, func_runs, motpar

# ---------------------------------------------------------------------------------

def add_motion_regressors_infocoupl(subj_info, univar_task, 
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
            subj_info[run].regressors += read_motion_par(motpar, univar_task, run) # list of 6 columns
            
    return subj_info, func_runs

# ---------------------------------------------------------------------------------

def main():
    
    subjlist = ['sub-{:03d}'.format(i) for i in range(1, 36)]
    #subjlist = ['sub-001']
    roilist = ['ba-17-18_contr-objscrvsbas_top-500']
    
    # ------------------------------------------------------
    # Utilities
    # ------------------------------------------------------
    
    # Identity interface
    subjinfo = Node(IdentityInterface(fields=['sub', 'roi']), name='subjinfo')
    subjinfo.iterables = [('sub', subjlist), ('roi', roilist)]
    
    # Datasink
    datasink = Node(nio.DataSink(parameterization=True), name='datasink')
    outdir = os.path.join(project_dir, 'info_coupling')
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    datasink.inputs.base_directory = outdir
    subs = [('_sub_', '_'), ('_roi_', '')]
    datasink.inputs.substitutions = subs
    
    # ------------------------------------------------------
    # Custom nodes
    # ------------------------------------------------------
    
    getfiles = Node(Function(input_names = ['sub', 'roi',
                                            'task', 'model',
                                            'approach', 'preproc'],
                             output_names = ['sub', 'roi',
                                             'task', 'model', 
                                             'approach', 
                                             'func_runs', 'motpar',
                                             'univar_task'], 
                             function = get_func_files),
                    name='getfiles')
    getfiles.inputs.approach = 'traintest'
    getfiles.inputs.task = ('train', 'test')
    getfiles.inputs.model = (5, 15)
    getfiles.inputs.preproc = 'smooth'
    
    decode_tc = Node(Function(input_names=['sub', 'roi', 'task', 'model', 
                                           'approach', 'dataformat',
                                           'func_runs', 'motpar'],
                              output_names=['allres',
                                            'func_runs', 'motpar'],
                              function = decode_timecourses),
                     name = 'decode_timecourses')
    decode_tc.inputs.dataformat = 'TRs'
    
    add_regressor = Node(Function(input_names=['tc', 'func_runs', 'motpar'],
                                  output_names=['subj_info', 'func_runs', 'motpar'],
                                  function = add_evidence_regressor),
                         name = 'add_regressor')
    
    add_motion_reg = Node(Function(input_names=['subj_info', 'univar_task',
                                                'func_runs', 'motpar',
                                                'use_motion_reg'],
                                   output_names=['subj_info', 'func_runs'],
                                   function=add_motion_regressors_infocoupl),
                          name='add_motion_reg')
    add_motion_reg.inputs.use_motion_reg = True
    
    # ------------------------------------------------------
    # SPM functions
    # ------------------------------------------------------
    
    # Node that defines the SPM model
    spmmodel = Node(modelgen.SpecifySPMModel(), name='spmmodel')
    spmmodel.inputs.high_pass_filter_cutoff = 128.
    spmmodel.inputs.concatenate_runs = False
    spmmodel.inputs.input_units = 'secs'
    spmmodel.inputs.output_units = 'secs'
    spmmodel.inputs.time_repetition = 1.0
    
    # Inputs:
    # - subj_info
    # - functional_runs
    
    # ------------------------------------------------------
    
    # Level 1 design node
    level1design = Node(Level1Design(), name='level1design')
    level1design.inputs.timing_units = 'secs'
    level1design.inputs.interscan_interval = 1.0
    level1design.inputs.bases = {'hrf':{'derivs': [0,0]}}
    level1design.inputs.flags = {'mthresh': 0.8}
    level1design.inputs.microtime_onset = 6.0
    level1design.inputs.microtime_resolution = 11
    level1design.inputs.model_serial_correlations = 'AR(1)'
    level1design.inputs.volterra_expansion_order = 1

    # Inputs:
    # - session_info (from SpecifySPMModel)
    
    # ------------------------------------------------------
    
    # Estimate model node
    modelest = Node(EstimateModel(), name='modelest')
    modelest.inputs.estimation_method = {'Classical': 1}
    modelest.inputs.write_residuals = False
    
    # Inputs:
    # - SPM.mat file (from Level1Design)
    
    # ------------------------------------------------------
    
    contrasts = []
    contrasts.append(('task>baseline', 'T', ['expected', 'unexpected'],
                     [0.5, 0.5]))
    contrasts.append(('exp>unexp', 'T', ['expected', 'unexpected'],
                     [1., -1.]))
    contrasts.append(('unexp>exp', 'T', ['expected', 'unexpected'],
                      [-1., 1.]))
    contrasts.append(('exp>baseline', 'T', ['expected'], [1.0]))
    contrasts.append(('unexp>baseline', 'T', ['unexpected'], [1.0]))
    
    # Estimate contrast node
    contrest = Node(EstimateContrast(), name='contrest')
    contrest.inputs.contrasts = contrasts
    
    # Inputs:
    # - Beta files
    # - SPM.mat file
    # - Contrasts
    # - Residual image
    
    # ------------------------------------------------------
    # Workflow
    # ------------------------------------------------------
    
    infocoupl_wf = Workflow(name='infocoupl_wf')
    infocoupl_wf.base_dir = project_dir
    tobeconnected = [(subjinfo, getfiles, [('sub', 'sub'), ('roi', 'roi')]),
                     (getfiles, decode_tc, [('sub', 'sub'), ('roi', 'roi'),
                                            ('task', 'task'), ('model', 'model'), 
                                            ('approach', 'approach'),
                                            ('func_runs', 'func_runs'),
                                            ('motpar', 'motpar')]),
                     (getfiles, add_motion_reg, [('univar_task', 'univar_task')]),
                     (decode_tc, add_regressor, [('allres', 'tc'),
                                                 ('func_runs', 'func_runs'),
                                                 ('motpar', 'motpar')]),
                     (add_regressor, add_motion_reg, [('subj_info', 'subj_info'),
                                                      ('func_runs', 'func_runs'),
                                                      ('motpar', 'motpar')]),
                     (add_motion_reg, spmmodel, [('subj_info', 'subject_info'),
                                                 ('func_runs', 'functional_runs')]),
                     (spmmodel, level1design, [('session_info', 'session_info')]),
                     (level1design, modelest, [('spm_mat_file', 'spm_mat_file')]),
                     (modelest, datasink, [('beta_images', 'betas'),
                                           ('spm_mat_file', 'betas.@a'),
                                           ('residual_image', 'betas.@b')]),
                     (modelest, contrest, [('spm_mat_file', 'spm_mat_file')]),
                     (modelest, contrest, [('beta_images', 'beta_images')]),
                     (modelest, contrest, [('residual_image', 'residual_image')]),
                     (contrest, datasink, [('con_images', 'contrasts'),
                                    ('spmT_images', 'contrasts.@a'),
                                    ('spm_mat_file', 'contrasts.@b')])
                     ]
    infocoupl_wf.connect(tobeconnected)
    
    # Draw workflow
    infocoupl_wf.write_graph(graph2use='orig', dotfilename='./workflow_graphs/graph_infocoupl.dot')
    
    infocoupl_wf.config['execution']['poll_sleep_duration'] = 1
    infocoupl_wf.config['execution']['job_finished_timeout'] = 120
    infocoupl_wf.config['execution']['remove_unnecessary_outputs'] = True
    infocoupl_wf.config['execution']['stop_on_first_crash'] = True

    infocoupl_wf.config['logging'] = {
            'log_directory': infocoupl_wf.base_dir+'/'+infocoupl_wf.name,
            'log_to_file': False}

    # run using PBS:
    #infocoupl_wf.run()
    infocoupl_wf.run('PBS', plugin_args={'max_jobs' : 100,
                                         'qsub_args': '-l walltime=1:00:00,mem=16g',
                                         'max_tries':3,
                                         'retry_timeout': 5,
                                         'max_jobname_len': 15})
    
    
    
    
if __name__=="__main__":
    main()
    
    # sub='sub-001'
    # task=('train', 'test')
    # model=(5, 15)
    # roi='ba-17-18_L_contr-objscrvsbas_top-1000'
    # approach = 'traintest'
    
    # allres, _, _, _ = decode_timecourses(sub, roi, task, model, approach, dataformat='TRs',
    #                    func_runs=None, motpar=None)
    
    # allres.to_csv('example_timecourse_TRs.csv', index=False)    