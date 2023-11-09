from nipype import Node, Workflow, IdentityInterface, Function
from configs import project_dir

def decode_FIR_timecourses(sub, roi, task, model, approach):
    """
    """
    import os
    import numpy as np
    import pandas as pd
    import sys
    sys.path.append('/project/3018040.05/rotscenetask_fmri/analysis/')
    from mvpa.loading import load_betas
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
        
    max_delay = 9
    
    if approach=='CV':
        
        DS = load_betas(opt, mask_templ=mask_templ, fir=True,
                        max_delay=max_delay)
        
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
        
        # Just normal betas for training
        trainDS = load_betas(train_opt, mask_templ=mask_templ,
                                fir=False)
        
        if trainDS is not None:
            
            trainDS = correct_labels(trainDS, train_opt)
            
            testDS = load_betas(test_opt, mask_templ=mask_templ, 
                                fir=True, max_delay=max_delay)
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
        
        return allres, opt
    
    else:
        return np.nan, opt

# ---------------------------------------------------------------------------------

def correlate_timeseqs(tc, opt):
    import pandas as pd
    import numpy as np
    import sys
    sys.path.append('/project/3018040.05/rotscenetask_fmri/analysis/')
    from mvpa.loading import load_betas
    from mvpa.mvpa_utils import split_expunexp
    from utils import Options
    from nilearn.image import new_img_like
    
    n_timepoints = tc.delay.nunique()
    tc = tc.groupby(['delay', 'expected']).mean().reset_index()
    
    # load FIR timecourses
    univar_opt = Options(
        sub=opt.sub, 
        task='test',
        model=27
    )
    
    wholebrainDS = load_betas(univar_opt, mask_templ=None, 
                             fir=True)
    n_voxels = wholebrainDS.samples.shape[1] # before removing NaNs
    wholebrainDS = split_expunexp(wholebrainDS)
    nanmask = np.all(np.isfinite(wholebrainDS.samples), axis=0)
    wholebrainDS = wholebrainDS[:, nanmask]
    
    univar_df = pd.DataFrame(
        {'delay': wholebrainDS.sa.delay,
         'expected': wholebrainDS.sa.expected,
         'samples': list(wholebrainDS.samples)}
    )
    univar_df = univar_df.groupby(['delay', 'expected']).mean().reset_index()
    
    # Get (n. voxels x n. timepoints) arrays for exp and unexp
    exp_univar_array = np.vstack(univar_df[univar_df.expected==1].samples).T
    unexp_univar_array = np.vstack(univar_df[univar_df.expected==0].samples).T
    # Normalize
    exp_univar_array = (exp_univar_array - np.mean(exp_univar_array, axis=1, keepdims=True))/np.std(exp_univar_array, axis=1, keepdims=True)
    unexp_univar_array = (unexp_univar_array - np.mean(unexp_univar_array, axis=1, keepdims=True))/np.std(unexp_univar_array, axis=1, keepdims=True)
    
    # Same thing for multivariate sequence
    exp_multivar_array = np.hstack(tc[tc.expected==True].distance).reshape(1, n_timepoints)
    unexp_multivar_array = np.hstack(tc[tc.expected==False].distance).reshape(1, n_timepoints)
    exp_multivar_array = (exp_multivar_array - np.mean(exp_multivar_array, axis=1, keepdims=True))/np.std(exp_multivar_array, axis=1, keepdims=True)
    unexp_multivar_array = (unexp_multivar_array - np.mean(unexp_multivar_array, axis=1, keepdims=True))/np.std(unexp_multivar_array, axis=1, keepdims=True)
    
    # Compute Pearsons correlations
    exp_corrs = np.dot(exp_univar_array, exp_multivar_array.T)/(n_timepoints-1)
    unexp_corrs = np.dot(unexp_univar_array, unexp_multivar_array.T)/(n_timepoints-1)
    
    # Convert into brain maps
    i, j, k = wholebrainDS.fa.voxel_indices.T
    
    exp_map = np.full(wholebrainDS.a.voxel_dim, np.nan)
    exp_map[i, j, k] = exp_corrs.flatten()
    exp_map = new_img_like('/project/3018040.05/anat_roi_masks/wholebrain.nii', exp_map)
    
    unexp_map = np.full(wholebrainDS.a.voxel_dim, np.nan)
    unexp_map[i, j, k] = unexp_corrs.flatten()
    unexp_map = new_img_like('/project/3018040.05/anat_roi_masks/wholebrain.nii', unexp_map)
    
    return exp_map, unexp_map, opt
            
# ---------------------------------------------------------------------------------

def save_corrmaps(exp_map, unexp_map, opt):
    import nibabel as nb
    import os
    import pdb
    
    outdir = os.path.join('/project/3018040.05/',
                          'FIR_correlations')
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    
    try:
        nb.save(exp_map, os.path.join(outdir,
                                    opt.roi,
                                    f'{opt.sub}_exp.nii'))
        nb.save(unexp_map, os.path.join(outdir,
                                    opt.roi,
                                    f'{opt.sub}_unexp.nii'))
    except:
        pdb.set_trace()
        
    return    

# ---------------------------------------------------------------------------------

def main():
    
    #subjlist = ['sub-{:03d}'.format(i) for i in range(1, 36)]
    subjlist = ['sub-001']
    roilist = ['ba-17-18_contr-objscrvsbas_top-500']
    
    # ------------------------------------------------------
    # Utilities
    # ------------------------------------------------------
    
    # Identity interface
    subjinfo = Node(IdentityInterface(fields=['sub', 'roi']), name='subjinfo')
    subjinfo.iterables = [('sub', subjlist), ('roi', roilist)]
    
    # ------------------------------------------------------
    # Custom nodes
    # ------------------------------------------------------
    
    decode_tc = Node(Function(input_names = ['sub', 'roi',
                                             'task', 'model',
                                             'approach'],
                              output_names = ['allres', 'opt'],
                              function = decode_FIR_timecourses),
                     name='decode_FIRs')
    decode_tc.inputs.approach = 'traintest'
    decode_tc.inputs.task = ('train', 'test')
    decode_tc.inputs.model = (5, 24)
    
    correlate_node = Node(Function(input_names = ['tc', 'opt'],
                                   output_names = ['exp_map', 'unexp_map', 'opt'],
                                   function = correlate_timeseqs),
                          name='correlate_timeseqs')
    
    saving_node = Node(Function(input_names = ['exp_map', 'unexp_map', 'opt'],
                                output_names = [],
                                function = save_corrmaps),
                       name='save_corrmaps')
    
    # ------------------------------------------------------
    # Workflow
    # ------------------------------------------------------
    
    infoFIR_wf = Workflow(name='info_FIR_wf')
    infoFIR_wf.base_dir = project_dir
    
    tobeconnected = [(subjinfo, decode_tc, [('sub', 'sub'),
                                            ('roi', 'roi')]),
                     (decode_tc, correlate_node, [('allres', 'tc'),
                                                  ('opt', 'opt')]),
                     (correlate_node, saving_node, [('exp_map', 'exp_map'),
                                                    ('unexp_map', 'unexp_map'),
                                                    ('opt', 'opt')])]
    infoFIR_wf.connect(tobeconnected)
    
    infoFIR_wf.write_graph(graph2use='orig', dotfilename='./workflow_graphs/graph_infoFIR.dot')
    
    infoFIR_wf.config['execution']['poll_sleep_duration'] = 1
    infoFIR_wf.config['execution']['job_finished_timeout'] = 120
    infoFIR_wf.config['execution']['remove_unnecessary_outputs'] = True
    infoFIR_wf.config['execution']['stop_on_first_crash'] = True

    infoFIR_wf.config['logging'] = {
            'log_directory': infoFIR_wf.base_dir+'/'+infoFIR_wf.name,
            'log_to_file': False}
    
    infoFIR_wf.run()
    # infoFIR_wf.run('PBS', plugin_args={'max_jobs' : 200,
    #                                     'qsub_args': '-l walltime=1:00:00,mem=16g',
    #                                     'max_tries':3,
    #                                     'retry_timeout': 5,
    #                                     'max_jobname_len': 15})
    
    
if __name__ == "__main__":
    main()