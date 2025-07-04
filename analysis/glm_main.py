from nipype import Node, Workflow, IdentityInterface, Function
from nipype.algorithms import modelgen
from nipype.interfaces import spm
from nipype.interfaces.spm.model import Level1Design, EstimateModel, EstimateContrast
from nipype.interfaces.base import Bunch
import nipype.interfaces.io as nio
from nipype.interfaces.matlab import MatlabCommand
from configs import project_dir, spm_dir, matlab_cmd, bids_dir
MatlabCommand.set_default_matlab_cmd(matlab_cmd)
MatlabCommand.set_default_paths(spm_dir)
from glm.motionparameters import add_motion_regressors
import pandas as pd

# ---------------------------------------------------------------------------------

def modelspecify(evfileslist, model, motpar):
    """
    Inputs:
    - evfileslist: list of events .tsv files, one per run
    """
    import pandas as pd
    import sys
    sys.path.append('/project/3018040.05/rotscenetask_fmri/analysis/')
    from glm.modelspec import (
        specify_model_funcloc, 
        specify_model_train,
        specify_model_test
        )
    
    task = evfileslist[0].split('_')[1].split('task-')[1]
    
    if task=='funcloc':
        subj_info = [specify_model_funcloc(ef, model) for ef in evfileslist] 
    elif task=='train':
        subj_info = [specify_model_train(ef, model) for ef in evfileslist]
    elif task=='test':
        behav = pd.read_csv(evfileslist[0].replace('_run-1_events', '_beh'), sep='\t')
        subj_info = [specify_model_test(ef, model, behav) for ef in evfileslist]
    else:
        raise Exception('"{:s}" is not a known task.'.format(task))
    
    return subj_info, task, motpar

# ---------------------------------------------------------------------------------

def contrastspecify(spm_mat_file):
    
    task = spm_mat_file.split('/')[-4].split('_task_')[1]
    modelno = int(spm_mat_file.split('/')[-3].split('_model_')[1])
    
    contrasts = []
    
    if task=='funcloc':
        
        # conditions = ['faces', 'objects', 'scenes', 'scrambled', 'buttonpress']
        
        if modelno==1:
            
            contrasts.append(('obj>scramb', 'T', ['objects', 'scrambled'], [1., -1.]))
            contrasts.append(('scenes>objects', 'T', ['scenes', 'objects'], [1., -1.]))
            contrasts.append(('faces>scenes', 'T', ['faces', 'scenes'], [1., -1.]))
            contrasts.append(('faces>objects', 'T', ['faces', 'objects'], [1., -1.]))
            contrasts.append(('faces>obj+scenes', 'T', ['faces', 'objects', 'scenes'], [1., -0.5, -0.5]))
            contrasts.append(('scenes>obj+faces', 'T', ['scenes', 'objects', 'faces'], [1., -0.5, -0.5]))
            contrasts.append(('bpress>presentation', 'T', 
                              ['buttonpress', 'faces', 'objects', 'scenes', 'scrambled'], 
                              [1., -0.25, -0.25, -0.25, -0.25]))
        
        elif modelno==2:
            
            contrasts.append(('stimulus>baseline', 'T', ['stimulus', 'baseline'], [1., -1.]))
            
        elif modelno==3:
            
            contrasts.append(('objscr>baseline', 'T', ['objscr', 'baseline'], [1., -1.]))
            
        else:
            raise Exception('"{:d}" is not a known model.'.format(modelno))
    
    elif task=='train':
        
        if modelno==1:
            
            contrasts.append(('near>far', 'T', ['near', 'far'], [1., -1.]))
            contrasts.append(('far>near', 'T', ['far', 'near'], [1., -1.]))
            contrasts.append(('bpress>presentation', 'T', 
                              ['buttonpress', 'near', 'far'], [1., -0.5, -0.5]))
        
        elif modelno==2:
            
            contrasts.append(('wide>narrow', 'T', ['wide', 'narrow'], [1., -1.]))
            contrasts.append(('narrow>wide', 'T', ['narrow', 'wide'], [1., -1.]))
            
        else:
            raise Exception('"{:d}" is not a known model.'.format(modelno))
    
    elif task=='test':
        
        if modelno==2:
            
            contrasts.append(('near>far', 'T', ['near', 'far'], [1., -1.])) 
            contrasts.append(('far>near', 'T', ['far', 'near'], [1., -1.])) 
        
        elif modelno==3:
            
            contrasts.append(('init wide>init narrow', 'T', ['init_wide', 'init_narrow'], [1., -1.]))
        
        elif modelno==4:
            
            contrasts.append(('final wide>final narrow', 'T', ['final_wide', 'final_narrow'], [1., -1.]))
        
        elif modelno==5:
            
            contrasts.append(('unexpected>expected', 'T', ['unexpected', 'expected'], [1., -1.]))
            contrasts.append(('expected>unexpected', 'T', ['expected', 'unexpected'], [1., -1.]))
            
        elif modelno==9:
            
            contrasts.append(('unexpected>expected', 'T', 
                             ['unexpected', 'expected_1', 'expected_2', 'expected_3'],
                             [1., -0.333, -0.333, -0.333]))
            
        elif modelno==14:
            
            contrasts.append(('Exp30>Exp90', 'T', ['rot30_exp', 'rot90_unexp'], [1., -1.]))
            contrasts.append(('Unexp30>Unexp90', 'T', ['rot30_unexp', 'rot90_unexp'], [1., -1.]))
            contrasts.append(('Unexp30>Exp30', 'T', ['rot30_unexp', 'rot30_exp'], [1., -1.]))
            contrasts.append(('Unexp90>Exp90', 'T', ['rot90_unexp', 'rot90_exp'], [1., -1.]))
        
        else:
            raise Exception('"{:d}" is not a known model.'.format(modelno))
    
    return contrasts

# ---------------------------------------------------------------------------------
    
def main():
    
    use_FIR = False
    
    # Utilities
    
    # Identity interface
    subjlist = [f'sub-{i:03d}' for i in range(1, 36)]
    #subjlist = ['sub-010']
    
    subjinfo = Node(IdentityInterface(fields=['sub']), name='subjinfo')
    subjinfo.iterables = [('sub', subjlist)]
    
    # Datagrabber to get event files
    get_events = Node(nio.DataGrabber(infields=['sub', 'task', 'preproc'],
                                      outfields=['events', 'func_runs', 'motpar']),
                      name='get_events')
    get_events.inputs.base_directory = bids_dir
    
    # Necessary default parameters
    get_events.inputs.template = '*' 
    get_events.inputs.sort_filelist = True
    
    # Templates
    get_events.inputs.template_args = {'events': [['sub', 'task']],
                                   'func_runs': [['sub', 'preproc', 'task']],
                                   'motpar': [['sub']]}
    get_events.inputs.field_template = {'events': '%s/func/*_task-%s_*_events.tsv',
                                        'func_runs': 'derivatives/spm-preproc/%s/%s/*_task-%s_*_bold.nii',
                                        'motpar': 'derivatives/spm-preproc/%s/realign_unwarp/rp_*.txt'}

    get_events.inputs.preproc = 'smooth'

    get_events.iterables = [('task', ['test'])]
    
    # Datasink
    datasink = Node(nio.DataSink(parameterization=True), name='datasink')
    datasink.inputs.base_directory = '/project/3018040.05/bids/derivatives/spm-preproc/derivatives/spm-stats'
    subs = [('_sub_', ''), ('_task_', ''), ('_model', 'model'), 
            ('_use_motion', 'use_motion'), ('_bases_hrfderivs0.0', ''),
            ('_bases_firlength20.order10', 'FIR')]
    datasink.inputs.substitutions = subs 
    
    # Custom nodes
    spec_model = Node(Function(input_names = ['evfileslist', 'model', 'motpar', 'motion_reg'],
                            output_names = ['subj_info', 'task', 'motpar'],
                            function=modelspecify), name='spec_model')

    spec_model.itersource = ('get_events', 'task')
    spec_model.iterables = [('model', {'test': [38]})]
    
    add_motion_reg = Node(Function(input_names=['subj_info', 'task', 'use_motion_reg', 'motpar'],
                               output_names=['subj_info'],
                               function=add_motion_regressors), name='add_motion_reg')
    add_motion_reg.inputs.use_motion_reg = True
    
    spec_contrast = Node(Function(input_names = ['spm_mat_file'],
                              output_names = ['contrasts'],
                              function=contrastspecify), name='spec_contrast')
    
    # Node that defines the SPM model
    # Inputs:
    # - subj_info
    # - functional_runs

    spmmodel = Node(modelgen.SpecifySPMModel(), name='spmmodel')
    spmmodel.inputs.high_pass_filter_cutoff = 128.
    spmmodel.inputs.concatenate_runs = False
    spmmodel.inputs.input_units = 'secs'
    spmmodel.inputs.output_units = 'secs'
    spmmodel.inputs.time_repetition = 1.0

    # Level 1 design node
    # Inputs:
    # - session_info (from SpecifySPMModel)
    
    level1design = Node(Level1Design(), name='level1design')
    level1design.inputs.timing_units = 'secs'
    level1design.inputs.interscan_interval = 1.0
    if use_FIR:
        level1design.iterables  = [('bases', [{'fir': {'length': 20, 'order': 10}}])]
    else:
        level1design.iterables = [('bases', [{'hrf': {'derivs': [0,0]}}])]
    level1design.inputs.flags = {'mthresh': 0.8}
    level1design.inputs.microtime_onset = 6.0
    level1design.inputs.microtime_resolution = 11
    level1design.inputs.model_serial_correlations = 'AR(1)'
    level1design.inputs.volterra_expansion_order = 1
    
    # Estimate model node
    # Inputs:
    # - SPM.mat file (from Level1Design)
    
    modelest = Node(EstimateModel(), name='modelest')
    modelest.inputs.estimation_method = {'Classical': 1}
    modelest.inputs.write_residuals = False
    
    # Estimate contrast node
    # Inputs:
    # - Beta files
    # - SPM.mat file
    # - Contrasts
    # - Residual image
    
    contrest = Node(EstimateContrast(), name='contrest')
    
    # Create workflow
    model_wf = Workflow(name='model_wf')
    model_wf.base_dir = '/project/3018040.05'
    tobeconnected = [(subjinfo, get_events, [('sub', 'sub')]),
                    (get_events, spec_model, [('events', 'evfileslist')]),
                    (get_events, spec_model, [('motpar', 'motpar')]),
                    (get_events, spmmodel, [('func_runs', 'functional_runs')]),
                    (spec_model, add_motion_reg, [('subj_info', 'subj_info')]),
                    (spec_model, add_motion_reg, [('motpar', 'motpar')]),
                    (spec_model, add_motion_reg, [('task', 'task')]),
                    (add_motion_reg, spmmodel, [('subj_info', 'subject_info')]),
                    (spmmodel, level1design, [('session_info', 'session_info')]),
                    (level1design, modelest, [('spm_mat_file', 'spm_mat_file')]),
                    (modelest, datasink, [('beta_images', 'betas'),
                                            ('spm_mat_file', 'betas.@a'),
                                            ('residual_image', 'betas.@b')])]

    docontrasts = False
    if docontrasts:
        tobeconnected += [(modelest, spec_contrast, [('spm_mat_file', 'spm_mat_file')]),
                        (modelest, contrest, [('spm_mat_file', 'spm_mat_file')]),
                        (modelest, contrest, [('beta_images', 'beta_images')]),
                        (modelest, contrest, [('residual_image', 'residual_image')]),
                        (spec_contrast, contrest, [('contrasts', 'contrasts')]),
                        (contrest, datasink, [('con_images', 'contrasts'),
                                        ('spmT_images', 'contrasts.@a'),
                                        ('spm_mat_file', 'contrasts.@b')])]

    model_wf.connect(tobeconnected)
    
    model_wf.config['execution']['poll_sleep_duration'] = 1
    model_wf.config['execution']['job_finished_timeout'] = 120
    model_wf.config['execution']['remove_unnecessary_outputs'] = True
    model_wf.config['execution']['stop_on_first_crash'] = True

    model_wf.config['logging'] = {
            'log_directory': model_wf.base_dir+'/'+model_wf.name,
            'log_to_file': False}

    # run using PBS:
    #model_wf.run()
    model_wf.run('PBS', plugin_args={'max_jobs' : 100, 'qsub_args': '-l nodes=1:ppn=1,walltime=1:00:00,mem=16g', 'max_tries':3,'retry_timeout': 5, 'max_jobname_len': 15})
    
if __name__=="__main__":
    main()
