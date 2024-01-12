import argparse
import os
from configs import project_dir, bids_dir

def get_roi_path(roi):
    if 'contr' in roi: # functional contrast
        roi_basedir = os.path.join(bids_dir, 'derivatives', 'spm-preproc', 
                                   'derivatives', 'roi-masks')
        mask_templ = os.path.join(roi_basedir, '{:s}/{:s}_' + roi + '.nii')
    else: # only anatomical map
        roi_basedir = os.path.join(project_dir, 'anat_roi_masks')
        mask_templ = os.path.join(roi_basedir, roi + '.nii')
    
    return mask_templ

# ---------------------------------------------------------------------------------

def pick_runs(n_samples, max_run, sub, negative=False):
    import random
    random.seed(sub)
    
    # randomly select 'sample_runs' runs
    chosenruns = random.sample(range(1, max_run+1), n_samples)
    if negative:
        chosenruns = [r for r in range(1, max_run+1) if r not in chosenruns]
    
    return chosenruns

# ---------------------------------------------------------------------------------

def decode_FIR_timecourses(sub, roi, task, model, approach, sample_runs=None, test_runs=False):
    """
    """
    import numpy as np
    import pandas as pd
    import sys
    sys.path.append('/project/3018040.05/rotscenetask_fmri/analysis/')
    from mvpa.loading import load_betas
    from mvpa.decoding import decode_CV, decode_traintest
    from mvpa.mvpa_utils import correct_labels
    from utils import Options, split_options
    import random
    random.seed(sub)
    
    opt = Options(
        sub=sub,
        roi=roi,
        task=task,
        model=model
    )
    
    mask_templ = get_roi_path(opt.roi)
        
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
            
            if sample_runs is not None:
                # randomly select 'sample_runs' runs
                chosenruns = pick_runs(sample_runs, testDS.chunks.max(), sub, negative=test_runs)
                testDS = testDS[np.isin(testDS.chunks, chosenruns)]
            else:
                chosenruns = None
                
            
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
        allres['chosenruns'] = ''.join((str(c)+',' if i != len(chosenruns)-1 else 
                                        str(c) for i, c in enumerate(chosenruns)))
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
        
        return allres, sub, roi, chosenruns
    
    else:
        return np.nan, sub, roi, chosenruns

# ---------------------------------------------------------------------------------

def save_timeseqs(tc, sub, roi, chosenruns=None):
    import os
    outdir = os.path.join('/project/3018040.05/',
                          'FIR_timeseries', 'decoding',
                          'test_m29')
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
        
    filename = f'{sub}_{roi}'
    if chosenruns is not None:
        filename += '_runs'
        for r in chosenruns:
            filename += f'-{r}'
    else:
        filename += '_allruns'
    filename += '.csv'
       
    tc.to_csv(os.path.join(outdir, filename), index=False)
    
    return

# ---------------------------------------------------------------------------------

def load_univar_ts(sub, task, model, src_roi=None):
    import pandas as pd
    import numpy as np
    import os
    import sys
    sys.path.append('/project/3018040.05/rotscenetask_fmri/analysis/')
    from mvpa.loading import load_betas
    from mvpa.mvpa_utils import split_expunexp
    from utils import Options
    
    if src_roi is not None:
        mask_templ = get_roi_path(src_roi)
    else:
        mask_templ = src_roi
    
    univar_opt = Options(
        sub=sub, 
        task=task,
        model=model
    )
    
    univarDS = load_betas(univar_opt, mask_templ=mask_templ,
                          fir=True)
    
    univarDS = split_expunexp(univarDS)
    nanmask = np.all(np.isfinite(univarDS.samples), axis=0)
    univarDS = univarDS[:, nanmask]
    
    univar_df = pd.DataFrame(
        {'delay': list(univarDS.sa.delay),
         'expected': list(univarDS.sa.expected),
         'samples': list(univarDS.samples),
         'run': list(univarDS.chunks)}
    )
    univar_df['subject'] = sub
    
    return univar_df, univarDS

# ---------------------------------------------------------------------------------

def save_univar_ts(univar_df, sub, roi):
    import os
    outdir = os.path.join('/project/3018040.05/',
                          'FIR_timeseries', 'univariate',
                          'test_m30')
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    univar_df.to_pickle(os.path.join(outdir, f'{sub}_{roi}.pkl'))
    
    return

# ---------------------------------------------------------------------------------

def correlate_timeseqs(tc, sub, roi, sample_runs=None, test_runs=False):
    import numpy as np
    import sys
    sys.path.append('/project/3018040.05/rotscenetask_fmri/analysis/')
    from nilearn.image import new_img_like
    
    
    n_timepoints = tc.delay.nunique()
    tc = tc.groupby(['delay', 'expected']).mean().reset_index()
    
    univar_df, wholebrainDS = load_univar_ts(sub, 'test', 30, chosenruns, src_roi=None)
    
    if sample_runs is not None:
        chosenruns = pick_runs(sample_runs, univar_df.run.max(), sub, negative=test_runs)
        univar_df = univar_df[univar_df['run'].isin(chosenruns)]
    
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
    
    return exp_map, unexp_map, sub, roi, chosenruns
            
# ---------------------------------------------------------------------------------

def save_corrmaps(exp_map, unexp_map, sub, roi, chosenruns):
    import nibabel as nb
    import os
    
    outdir = os.path.join('/project/3018040.05/',
                          'FIR_correlations', 'test_m29', roi)
    
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    
    filename = sub
    if chosenruns is not None:
        filename += '_runs'
        for r in chosenruns:
            filename += f'-{r}'
    else:
        filename += '_allruns'
    
    nb.save(exp_map, os.path.join(outdir,
                                filename+'_exp.nii'))
    nb.save(unexp_map, os.path.join(outdir,
                                filename+'_unexp.nii'))
    
    return    

# ---------------------------------------------------------------------------------

def granger_timeseqs(tc, sub, roi, chosenruns, src_roi):
    import numpy as np
    from statsmodels.tsa.stattools import grangercausalitytests
    import sys
    sys.path.append('/project/3018040.05/rotscenetask_fmri/analysis/')
    
    tc = tc.groupby(['delay', 'expected']).mean().reset_index()
    
    univar_df, _ = load_univar_ts(sub, 'test', 30, chosenruns, src_roi=src_roi)
    
    # Compute Granger causality tests:
    uv_ts_exp = np.mean(np.vstack(univar_df[univar_df.expected==True]['samples']), axis=1)
    uv_ts_unexp = np.mean(np.vstack(univar_df[univar_df.expected==False]['samples']), axis=1)
    mv_ts_exp = np.vstack(tc[tc.expected==True].distance)
    mv_ts_unexp = np.vstack(tc[tc.expected==False].distance)
    
    # "Feedback" univariate --> multivariate
    ts_uv2mv_exp = np.hstack([uv_ts_exp, mv_ts_exp])
    ts_uv2mv_unexp = np.hstack([uv_ts_unexp, mv_ts_unexp])
    
    gc_uv2mv_exp = grangercausalitytests(ts_uv2mv_exp, 1, verbose=False)
    gc_uv2mv_unexp = grangercausalitytests(ts_uv2mv_unexp, 1, verbose=False)
    
    # "Feedforward" multivariate --> univariate
    ts_mv2uv_exp = np.hstack([mv_ts_exp, uv_ts_exp])
    ts_mv2uv_unexp = np.hstack([mv_ts_unexp, uv_ts_unexp])
    
    gc_mv2uv_exp = grangercausalitytests(ts_mv2uv_exp, 1, verbose=False)
    gc_mv2uv_unexp = grangercausalitytests(ts_mv2uv_unexp, 1, verbose=False)
    
    # Compute F-statistic difference between feedback and feedforward
    f_diff_exp = gc_uv2mv_exp[1][0]['ssr_ftest'][0] - gc_mv2uv_exp[1][0]['ssr_ftest'][0]
    f_diff_unexp = gc_uv2mv_unexp[1][0]['ssr_ftest'][0] - gc_mv2uv_unexp[1][0]['ssr_ftest'][0]

# ---------------------------------------------------------------------------------

def main(sub, roi):
    print('--------------------------------')
    print(f'Subject: {sub}, ROI: {roi}')
    print('--------------------------------')
    
    print('Starting decoding...')
    allres, sub, roi, chosenruns = decode_FIR_timecourses(sub, roi, 
                                              ('train', 'test'),
                                              (5, 29), 'traintest',
                                              sample_runs=5, test_runs=True)
    print('Done! Saving timeseries files...')
    save_timeseqs(allres, sub, roi, chosenruns=chosenruns)
    print('Saved multivariate timeseries files.')
    print('Loading univariate sequences...')
    univar_df, _ = load_univar_ts(sub, 'test', 30, src_roi='glasser-v5_R')
    save_univar_ts(univar_df, sub, roi)
    print('Saved univariate timeseries files.')
    # print('Done! Computing correlations...')
    # exp_map, unexp_map, sub, roi, chosenruns = correlate_timeseqs(allres, sub, roi, chosenruns)
    # print('Done!')
    # save_corrmaps(exp_map, unexp_map, sub, roi, chosenruns)
    # print('Saved correlation map files.')
    return

# ---------------------------------------------------------------------------------

if __name__=="__main__":
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--sub", required=True, type=str, help="Subject")
    parser.add_argument("--roi", required=True, type=str, help="ROI")

    # Parse arguments
    args = parser.parse_args()

    # Call the main function with the args namespace
    main(args.sub, args.roi)