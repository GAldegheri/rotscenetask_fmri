from mvpa2.base import dataset
from mvpa2.mappers.fx import FxMapper
import numpy as np
from .loading import load_TRs, load_betas, load_trialbetas
from .classify_models import DivideInThirds, isExpUnexp, \
    is30or90, isAorB, isWideNarrow, isAllViews
import warnings
import random
import ipdb

# =================================================================================================
# MVPA utilities
# =================================================================================================

def split_expunexp(DS):
    
    expunexp = []
    for t in DS.targets:
        if 'unexp' in t:
            expunexp.append(0)
        elif 'exp' in t:
            expunexp.append(1)
        else:
            expunexp.append(None)
    assert(len(DS.targets)==len(expunexp))
    DS.sa['expected'] = np.array(expunexp)
    
    return DS

# -------------------------------------------------------------------------------------------------

def divide_in_thirds(DS):
    
    trialsplits = []
    for t in DS.targets:
        if '_1' in t:
            trialsplits.append(1)
        elif '_2' in t:
            trialsplits.append(2)
        elif '_3' in t and not 'unexp' in t: # to distinguish it from _30
            trialsplits.append(3)
        else:
            trialsplits.append(None)
    assert(len(DS.targets)==len(trialsplits))
    DS.sa['trialsplit'] = np.array(trialsplits)
    
    return DS

# -------------------------------------------------------------------------------------------------

def combine_splits(res):
    """
    Given a results dataframe divided in three splits (of trials),
    combine them in the appropriate way for each column.
    """
    for s in sorted(res.split.unique()):
        thissplitlength = len(res[res['split']==s])
        res.loc[res['split']==s, 'sample'] = list(range(thissplitlength))
    
    return res.groupby('sample').mean().reset_index().drop(
        ['sample', 'split'], axis=1)

# -------------------------------------------------------------------------------------------------

def split_views(DS, opt):
    
    if (opt.task=='train' and opt.model==4) or (opt.task=='test' and opt.model in [12, 16, 20, 23]):
       
        DS_A = DS[np.core.defchararray.find(DS.sa.targets,'A')!=-1]
        DS_B = DS[np.core.defchararray.find(DS.sa.targets,'B')!=-1]
        return (DS_A, DS_B)
    
    elif (opt.task=='train' and opt.model==5) or (opt.task=='test' and opt.model in [15, 17, 21, 24]):
        
        DS_30 = DS[np.core.defchararray.find(DS.sa.targets,'30')!=-1]
        DS_90 = DS[np.core.defchararray.find(DS.sa.targets,'90')!=-1]
        return (DS_30, DS_90)
        
    else:
        raise Exception('Task: {:s}, Model: {:s} does not support separate view decoding!'.format(opt.task, opt.model))
    
# -------------------------------------------------------------------------------------------------

def correct_labels(DS, opt):
    '''
    Fix labels if necessary, for models:
    - train 1: near -> rot30
    - train 3: wide_blockX -> wide
    - test 2: near -> rot30
    - test 3: init_wide -> wide
    - test 4: final_wide -> wide
    - test 6: wide_exp -> wide
    - test 7: wide_exp_1 -> wide
    - test 8: exp_wide -> wide
    - test 13: rot30_exp_1 -> rot30
    - test 12/15: A30_exp_1 -> A30
    '''
    
    if isExpUnexp(opt):
        
        DS = split_expunexp(DS) # adds a field 'expected' to DS
    
    if DivideInThirds(opt):
            
        DS = divide_in_thirds(DS) # adds a field 'trialsplit' to DS
    
    # -----------------------------------------
    if isWideNarrow(opt):
        
        wideind = [i for i, item in enumerate(DS.sa.targets) if 'wide' in item]
        narrind = [i for i, item in enumerate(DS.sa.targets) if 'narrow' in item]
        DS.sa.targets[wideind] = 'wide'
        DS.sa.targets[narrind] = 'narrow'
        
    elif is30or90(opt):
        
        rot30ind = [i for i, item in enumerate(DS.sa.targets) if '30' in item or 'near' in item]
        rot90ind = [i for i, item in enumerate(DS.sa.targets) if '90' in item or 'far' in item]
        DS.sa.targets = np.empty(DS.sa.targets.shape, dtype='<U21')
        DS.sa.targets[rot30ind] = 'rot30'
        DS.sa.targets[rot90ind] = 'rot90'
        
    elif isAllViews(opt):
        
        A30ind = [i for i, item in enumerate(DS.sa.targets) if 'A' in item and '30' in item]
        A90ind = [i for i, item in enumerate(DS.sa.targets) if 'A' in item and '90' in item]
        B30ind = [i for i, item in enumerate(DS.sa.targets) if 'B' in item and '30' in item]
        B90ind = [i for i, item in enumerate(DS.sa.targets) if 'B' in item and '90' in item]
        DS.sa.targets = np.empty(DS.sa.targets.shape, dtype='<U21')
        DS.sa.targets[A30ind] = 'A30'
        DS.sa.targets[A90ind] = 'A90'
        DS.sa.targets[B30ind] = 'B30'
        DS.sa.targets[B90ind] = 'B90'
        
    elif isAorB(opt):
        
        Aind = [i for i, item in enumerate(DS.sa.targets) if 'A' in item]
        Bind = [i for i, item in enumerate(DS.sa.targets) if 'B' in item]
        DS.sa.targets = np.empty(DS.sa.targets.shape, dtype='<U21')
        DS.sa.targets[Aind] = 'A'
        DS.sa.targets[Bind] = 'B'
        
    return DS

# -------------------------------------------------------------------------------------------------

def assign_loadfun(dataformat):
    if dataformat=='TRs':
        loadfun = load_TRs
    elif dataformat=='betas':
        loadfun = load_betas
    elif dataformat=='trialbetas':
        loadfun = load_trialbetas
    return loadfun

# -------------------------------------------------------------------------------------------------

def average_thirds(DS):
    '''
    Takes as input the "expected" dataset, 
    and averages samples across the 3 trial splits.
    TODO: make this work with TRs as well!
    '''
    
    warnings.warn('''Warning! This function is equivalent to 
    estimating the betas with 3 times as many trials for expected! 
    Use with caution''')
    
    meanmapper = FxMapper('samples', np.mean, uattrs=['chunks']) # average by chunk
    
    if any('wide' in x for x in DS.sa.targets):
        narrDS = dataset.vstack([i for i in DS if 'narrow' in i.sa.targets[0]], a=0)
        narrDS.sa.targets[:] = 'exp_narrow'
        wideDS = dataset.vstack([i for i in DS if 'wide' in i.sa.targets[0]], a=0)
        wideDS.sa.targets[:] = 'exp_wide'

        narrDS = meanmapper(narrDS)
        wideDS = meanmapper(wideDS)

        return dataset.vstack([wideDS, narrDS], a=0)
    
    else: # only exp/unexp, no wide/narrow
        DS.sa.targets[:] = 'expected'
        
        return meanmapper(DS)
    
# -------------------------------------------------------------------------------------------------
    
def randpick_third(DS):
    '''
    Instead of averaging across the 3 trial splits,
    just randomly select one per run.
    '''
    
    if any('wide' in x for x in DS.sa.targets):
        widenarrow = True
    else:
        widenarrow = False
        
    if any('30' in x for x in DS.sa.targets):
        rot3090 = True
    else:
        rot3090 = False
    
    onethirdDS = []
    for ch in np.unique(DS.chunks):
        if widenarrow:
            thisnarr = DS[(DS.chunks==ch) & (np.core.defchararray.find(DS.targets,'narrow')!=-1)]
            thiswide = DS[(DS.chunks==ch) & (np.core.defchararray.find(DS.targets,'wide')!=-1)]
            thisnarr = thisnarr[random.randint(0, 2)]
            thiswide = thiswide[random.randint(0, 2)]
            onethirdDS.extend([thiswide, thisnarr])
        elif rot3090:
            this30 = DS[(DS.chunks==ch) & (np.core.defcharray.find(DS.targets,'30')!=-1)]
            this90 = DS[(DS.chunks==ch) & (np.core.defcharray.find(DS.targets,'90')!=-1)]
            this30 = this30[random.randint(0, 2)]
            this90 = this90[random.randint(0, 2)]
            onethirdDS.extend([this30, this90])
        else:
            thischunk = DS[DS.chunks==ch]
            thischunk = thischunk[random.randint(0, 2)]
            onethirdDS.append(thischunk)
    
    return dataset.vstack(onethirdDS, a=0)
