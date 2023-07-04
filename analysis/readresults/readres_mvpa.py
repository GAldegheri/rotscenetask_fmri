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
    all_results = all_results.replace(np.nan, 'none')
    return all_results

def get_subj_avg(results, avg_decodedirs=False):
    results = results.drop(['chunk'], axis=1)
    ind_vars = ['subject', 'roi', 'approach', 
                'traindataformat', 'testdataformat', 'traintask',
                'testtask', 'trainmodel', 'testmodel', 
                'hemi', 'contrast', 'nvoxels', 'expected']
    ind_vars = [i for i in ind_vars if i in results.columns]
    
    if avg_decodedirs:
        # remove traintask, trainmodel, testtask, testmodel...
        ind_vars = [i for i in ind_vars if 'train' not in i and 'test' not in i]
        groupedres = []
        taskmodelpairs = list(results[['traintask', 'trainmodel']].drop_duplicates().itertuples(index=False, name=None))
        for t, m in taskmodelpairs:
            thistm = results[((results['traintask']==t)&(results['trainmodel']==m))|\
                ((results['testtask']==t)&(results['testmodel']==m))]
            thesemodels = list(thistm.trainmodel.unique())
            thesetasks = list(thistm.traintask.unique())
            thistm = thistm.groupby(ind_vars, dropna=False).mean().reset_index()
            thistm['traintask'] = thesetasks[0]+'_'+thesetasks[1]
            thistm['testtask'] = thesetasks[0]+'_'+thesetasks[1]
            thistm['trainmodel'] = str(thesemodels[0])+'_'+str(thesemodels[1])
            thistm['testmodel'] = str(thesemodels[0])+'_'+str(thesemodels[1])
            groupedres.append(thistm)
        results = pd.concat(groupedres)
    else:
        results = results.groupby(ind_vars, dropna=False).mean().reset_index()
    
    if 'view' in results.columns:
        results = results.drop(['view'], axis=1)
    
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
                contrasts.append('none')
                nvoxels.append('none')
        else:
            hemispheres.append('LR')
            contrasts.append('none')
            nvoxels.append('none')
    
    results['roi'] = roinames
    results['hemi'] = hemispheres
    results['contrast'] = contrasts
    results['nvoxels'] = nvoxels
    
    return results

def fill_in_nvoxels(results):
    ind_vars = ['subject', 'roi', 'approach', 
                'traindataformat', 'testdataformat', 'traintask',
                'testtask', 'trainmodel', 'testmodel', 
                'hemi', 'contrast', 'expected']
    ind_vars = [i for i in ind_vars if i in results.columns]
    
    # get unique combinations of independent variables:
    varcombinations = [r.to_dict() for _, r in
                       results[ind_vars].drop_duplicates().iterrows()]
    for vc in varcombinations:
        for i, v in enumerate(vc):
            if i==0:
                thismask = results[v]==vc[v]
            else:
                thismask &= results[v]==vc[v]
        theseresults = results[thismask]
        #for nv in theseresults.nvoxels.unique():
    
    return
            
if __name__=="__main__":
    results = merge_results(['/project/3018040.05/MVPA_results/mainanalysis.csv'])
    avgres = get_subj_avg(results, avg_decodedirs=True)
    