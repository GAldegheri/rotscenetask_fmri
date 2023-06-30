import pandas as pd
import warnings

def merge_results(res_list):
    """
    Given list of results files,
    returns a dataframe merging all of them.
    """
    all_results = []
    for r in res_list:
        all_results.append(pd.read_csv(r))
    all_results = pd.concat(all_results)
    return all_results

def get_subj_avg(results, avg_decodedirs=False):
    results = results.drop(['chunk'], axis=1)
    ind_vars = ['subject', 'roi', 'approach', 
                'traindataformat', 'testdataformat', 'traintask',
                'testtask', 'trainmodel', 'testmodel', 
                'hemi', 'contrast', 'nvoxels']
    ind_vars = [i for i in ind_vars if i in results.columns]
    
    if avg_decodedirs:
        # remove traintask, trainmodel, testtask, testmodel...
        ind_vars = [i for i in ind_vars if 'train' not in i and 'test' not in i]
        groupedres = []
        taskmodelpairs = list(results[['traintask', 'trainmodel']].drop_duplicates().itertuples(index=False, name=None))
        for t, m in taskmodelpairs:
            thistm = results[((results['traintask']==t)&(results['trainmodel']==m))|\
                ((results['testtask']==t)&(results['testmodel']==m))]
            thistm = thistm.groupby(ind_vars).mean().reset_index()
            thistm['traintask'] = taskmodelpairs[0][0]+'_'+taskmodelpairs[1][0]
            thistm['testtask'] = taskmodelpairs[0][0]+'_'+taskmodelpairs[1][0]
            thistm['trainmodel'] = taskmodelpairs[0][1]+'_'+taskmodelpairs[1][1]
            thistm['testmodel'] = taskmodelpairs[0][1]+'_'+taskmodelpairs[1][1]
            groupedres.append(thistm)
    else:
        results = results.groupby(ind_vars).mean().reset_index()
    
    return results    

def parse_roi_info(results):
    """
    Info to be extracted from ROI string:
    - ROI name (e.g. BA 17, LOC, PPA)
    - Hemisphere (L/R/LR)
    - Contrast (e.g. Object vs. Scrambled)
    - N voxels (e.g. 100, 1000, 'all' [all significant],
                None [no selection])
    """
    if 'roi' not in results.columns:
        warnings.warn('No ROI information in this dataframe!')
        return results
    
    roinames = []
    hemispheres = []
    contrasts = []
    nvoxels = []
    
    for r in results.roi:
        allinfo = r.split('_')
        roinames.append(allinfo[0])
        if len(allinfo) > 1:
            if allinfo[1] in ['L', 'R']:
                hemispheres.append(allinfo[1])
                contrindx = 2
            else:
                hemispheres.append('LR')
                contrindx = 1
            if len(allinfo) > contrindx:
                contrasts.append(allinfo[contrindx].split('contr-')[1])
                if 'allsignif' in allinfo[contrindx+1]:
                    nvoxels.append('all')
                elif 'top-' in allinfo[contrindx+1]:
                    nvoxels.append(int(allinfo[contrindx+1].split('top-')[1]))
            else:
                contrasts.append(None)
                nvoxels.append(None)
        else:
            hemispheres.append('LR')
            contrasts.append(None)
            nvoxels.append(None)
    
    results['roi'] = roinames
    results['hemi'] = hemispheres
    results['contrast'] = contrasts
    results['nvoxels'] = nvoxels
    
    return results
            
            
    