from mvpa2.base import dataset
import numpy as np
from scipy.stats import pearsonr
import random
from .core_classifiers import trainandtest_sklearn, \
    CV_leaveoneout, SplitHalfCorr
from .mvpa_utils import randpick_third, combine_splits, split_views
from .classify_models import isWideNarrow, isExpUnexp, \
    isAllViews, isAorB, is30or90, DivideInThirds
import pandas as pd

# =================================================================================================
# Decoding wrapper functions
# To support train/test and cross-validation for separate views,
# separate thirds of the data, etc.
# =================================================================================================

def decode_thirds(expDS, opt, approach, otherDS=None, trainortest=None):
    ''' 
    - expDS: the expected dataset, or the expected/unexpected tuple for test model 9
    - otherDS: not used for crossval
    - opt:
        - task, model: only one because this receives as input either the train/test expected set, or CV
    - approach: 'CV', 'traintest' or 'splithalf'
    - trainortest: (only applicable if approach=='traintest') whether
      expDS (the one divided in 3) should be used as training or test dataset.             
    '''
    
    justonethird = False
    
    if approach=='traintest':
        
        if justonethird:
        
            expDS = randpick_third(expDS)
            if trainortest=='train':
                res_exp = trainandtest_sklearn(expDS, otherDS, zscore_data=True)
            elif trainortest=='test':
                res_exp = trainandtest_sklearn(otherDS, expDS, zscore_data=True)
        
        else:
        
            splitsres = []
            for n in range(1, 4):
                thisexpDS = expDS[(expDS.sa.trialsplit==n)|(expDS.sa.trialsplit==None)]
                if trainortest=='train':
                    thisres = trainandtest_sklearn(thisexpDS, otherDS, zscore_data=True)
                    thisres['split'] = n
                    splitsres.append(thisres)
                elif trainortest=='test':
                    thisres = trainandtest_sklearn(otherDS, thisexpDS, zscore_data=True)
                    thisres['split'] = n
                    splitsres.append(thisres)
            res_exp = combine_splits(pd.concat(splitsres))
    
    elif approach in ['CV', 'splithalf']:
        
        if approach=='CV':
            decodefun = lambda x: CV_leaveoneout(x, zscore_data=True)
        else:
            decodefun = lambda x: SplitHalfCorr(x)
        
        if justonethird:
            
            expDS = randpick_third(expDS)
            if isWideNarrow(opt) or is30or90(opt) or isAllViews(opt) or isAorB(opt): # test model 7/13/12/15
                res_exp = decodefun(expDS)
            else: # test model 9
                assert isinstance(expDS, tuple)
                expunexpDS = dataset.vstack([expDS[0], expDS[1]], a=0)
                res_exp = decodefun(expunexpDS)
        
        else:
            
            res_exp = []
            for n in range(1, 4):
                if isWideNarrow(opt) or is30or90(opt) or isAllViews(opt) or isAorB(opt): # test model 7/13/12/15
                    thisexpDS = expDS[(expDS.sa.trialsplit==n)|(expDS.sa.trialsplit==None)]
                    res_exp.append(decodefun(thisexpDS))
                else: # test model 9
                    assert isinstance(expDS, tuple)
                    thisexpDS = expDS[0][expDS[0].sa.trialsplit==n]
                    thisDS = dataset.vstack([thisexpDS, expDS[1]], a=0)
                    res_exp.append(decodefun(thisDS))

            res_exp = combine_splits(pd.concat(res_exp))
            
    return res_exp

# -------------------------------------------------------------------------------------------------

def decode_viewspecific(trainDS, testDS, trainopt, testopt, split=None, thirds='none'):
    '''
    split = 'train'/'test'/'both'
    thirds = 'train'/'test'/'none'
    '''
    
    if split=='train':
        
        trainDS = list(split_views(trainDS, trainopt))
        testDS = [testDS]
        
    elif split=='test':
        
        trainDS = [trainDS]
        testDS = list(split_views(testDS, testopt))
        
    elif split=='both':
       
        trainDS = list(split_views(trainDS, trainopt))
        testDS = list(split_views(testDS, testopt))
        
    else:
        raise Exception('Must specify which dataset to split! (train, test, or both)')
    
    res = []
    
    for i, tr in enumerate(trainDS):
        for j, te in enumerate(testDS):
            if i==j or split!='both': # match same view in train and test (e.g. A with A, B with B)
                if thirds=='train':
                    thisres = decode_thirds(tr, trainopt, 'traintest', \
                                             otherDS=te, \
                                             trainortest='train')
                elif thirds=='test':
                    thisres = decode_thirds(te, testopt, 'traintest', \
                                             otherDS=tr, \
                                             trainortest='test')
                elif thirds=='none':
                    thisres = trainandtest_sklearn(tr, te, zscore_data=True)
                
                thisres['view'] = i+1 if split=='train' else j+1 
                res.append(thisres)
    
    return pd.concat(res)
                
# -------------------------------------------------------------------------------------------------

def decode_traintest(trainDS, testDS, trainopt, testopt):
    
    '''
    No need to consider test model 9 here. 
    At the moment, not possible to train in a view-invariant model
    and test on a view-specific.
    '''
    
    if isExpUnexp(trainopt) and not isExpUnexp(testopt):
        
        trainDS_exp = trainDS[(trainDS.sa.expected==1)|(trainDS.sa.expected==None)]
        trainDS_unexp = trainDS[(trainDS.sa.expected==0)|(trainDS.sa.expected==None)]
        
        if DivideInThirds(trainopt): 
            # divide in thirds only train
            
            if isAllViews(trainopt) and isAllViews(testopt):
                
                # split both trainDS and testDS into views (A/B or 30/90)
                res_exp = decode_viewspecific(trainDS_exp, testDS, trainopt, testopt, \
                                                                   split='both', thirds='train')
                res_unexp = decode_viewspecific(trainDS_unexp, testDS, trainopt, testopt, \
                                                                       split='both', thirds='none')
                
            elif isAllViews(trainopt) and not isAllViews(testopt):
                    
                # only split trainDS into views
                res_exp = decode_viewspecific(trainDS_exp, testDS, trainopt, testopt, \
                                                                   split='train', thirds='train')
                res_unexp = decode_viewspecific(trainDS_unexp, testDS, trainopt, testopt, \
                                                                       split='train', thirds='none')
                
            elif not isAllViews(trainopt) and isAllViews(testopt):
                
                # only split testDS into views
                res_exp = decode_viewspecific(trainDS_exp, testDS, trainopt, testopt, \
                                                                   split='test', thirds='train')
                res_unexp = decode_viewspecific(trainDS_unexp, testDS, trainopt, testopt, \
                                                                       split='test', thirds='none')
            else: # no view-specific decoding
                
                res_exp = decode_thirds(trainDS_exp, trainopt, 'traintest', \
                                        otherDS=testDS, trainortest='train')
                
                res_unexp = trainandtest_sklearn(trainDS_unexp, testDS, zscore_data=True)
        
        #else: # exp/unexp without dividing in thirds - not really a thing
        res_exp['expected'] = True
        res_unexp['expected'] = False
        
        return pd.concat([res_exp, res_unexp])
    
    elif isExpUnexp(testopt) and not isExpUnexp(trainopt):
        
        testDS_exp = testDS[(testDS.sa.expected==1)|(testDS.sa.expected==None)]
        testDS_unexp = testDS[(testDS.sa.expected==0)|(testDS.sa.expected==None)]
        
        if DivideInThirds(testopt): #divide in thirds
            
            if isAllViews(trainopt) and isAllViews(testopt):
                
                # split both trainDS and testDS into views (A/B or 30/90)
                res_exp = decode_viewspecific(trainDS, testDS_exp, trainopt, testopt, \
                                                                   split='both', thirds='test')
                res_unexp = decode_viewspecific(trainDS, testDS_unexp, trainopt, testopt, \
                                                                       split='both', thirds='none')
                
            elif isAllViews(trainopt) and not isAllViews(testopt):
                    
                res_exp = decode_viewspecific(trainDS, testDS_exp, trainopt, testopt, \
                                                                   split='train', thirds='test')
                res_unexp = decode_viewspecific(trainDS, testDS_unexp, trainopt, testopt, \
                                                                       split='train', thirds='none')
                
            elif not isAllViews(trainopt) and isAllViews(testopt):
                
                res_exp = decode_viewspecific(trainDS, testDS_exp, trainopt, testopt, \
                                                                   split='test', thirds='test')
                res_unexp = decode_viewspecific(trainDS, testDS_unexp, trainopt, testopt, \
                                                                       split='test', thirds='none')
            else:
            
                res_exp = decode_thirds(testDS_exp, testopt, 'traintest', \
                                        otherDS=trainDS, trainortest='test')

                res_unexp = trainandtest_sklearn(trainDS, testDS_unexp, zscore_data=True)
         
        #else: # exp/unexp without dividing in thirds - not really a thing
        
        res_exp['expected'] = True
        res_unexp['expected'] = False
        
        return pd.concat([res_exp, res_unexp])
    
    elif not isExpUnexp(trainopt) and not isExpUnexp(testopt):
        # just standard decoding, no exp/unexp split, no third split
        
        if isAllViews(trainopt) and isAllViews(testopt):
            
            res = decode_viewspecific(trainDS, testDS, trainopt, testopt, \
                                                       split='both', thirds='none')
            
        elif isAllViews(trainopt) and not isAllViews(testopt): 
            
            res = decode_viewspecific(trainDS, testDS, trainopt, testopt, \
                                                       split='train', thirds='none')
            
        elif not isAllViews(trainopt) and isAllViews(testopt):
            
            res = decode_viewspecific(trainDS, testDS, trainopt, testopt, \
                                                       split='test', thirds='none')
            
        else:
            
            res = trainandtest_sklearn(trainDS, testDS, zscore_data=True)
        
        return res
    
# -------------------------------------------------------------------------------------------------

def traintest_dist(trainDS, testDS, trainopt, testopt):
    '''
    Only one use case (for now): testDS is expected/unexpected,
    with expected to be divided in 3. 
    Also test task/model needs to be view-specific.
    '''
    
    assert(isExpUnexp(testopt) and DivideInThirds(testopt) 
           and isAllViews(testopt) and isAllViews(trainopt))
    
    testDS_exp = testDS[testDS.sa.expected==1]
    testDS_unexp = testDS[testDS.sa.expected==0]
    
    dist_exp = 0
    for n in range(1, 4):
        thisexpDS = testDS_exp[(testDS_exp.sa.trialsplit==n)|(testDS_exp.sa.trialsplit==None)]
        dist_exp += distance_TrainTest(trainDS, thisexpDS)
    dist_exp /= 3
    
    dist_unexp = distance_TrainTest(trainDS, testDS_unexp)
    
    return dist_exp - dist_unexp

# -------------------------------------------------------------------------------------------------

def decode_CV(DS, opt):
    
    if isExpUnexp(opt):
        
        expDS = DS[(DS.sa.expected==1)|(DS.sa.expected==None)]
        unexpDS = DS[(DS.sa.expected==0)|(DS.sa.expected==None)]
        
        if DivideInThirds(opt):
            
            if isAllViews(opt):
                
                (expDS_1, expDS_2) = split_views(expDS, opt)
                (unexpDS_1, unexpDS_2) = split_views(unexpDS, opt)
                res_exp_1 = decode_thirds(expDS_1, opt, 'CV')
                res_exp_2 = decode_thirds(expDS_2, opt, 'CV')
                res_exp = pd.concat([res_exp_1, res_exp_2])
                
                res_unexp_1 = CV_leaveoneout(unexpDS_1, zscore_data=True)
                res_unexp_2 = CV_leaveoneout(unexpDS_2, zscore_data=True)
                res_unexp = pd.concat([res_unexp_1, res_unexp_2])
                
                res_exp['expected'] = True
                res_unexp['expected'] = False
                
                return pd.concat([res_exp, res_unexp])
            
            elif isWideNarrow(opt) or is30or90(opt) or isAorB(opt):
                
                res_exp = decode_thirds(expDS, opt, 'CV')
                res_unexp = CV_leaveoneout(unexpDS, zscore_data=True)
                
                res_exp['expected'] = True
                res_unexp['expected'] = False

                return pd.concat([res_exp, res_unexp])
            
            else: # test model 9
                
                res = decode_thirds((expDS, unexpDS), opt, 'CV')
                return res
        
        #else: (exp/unexp without dividing in thirds)
        
    else: # no expected vs. unexpected
        
        if isAllViews(opt):
            
            (DS_1, DS_2) = split_views(DS, opt)
            res_1 = CV_leaveoneout(DS_1, zscore_data=True)
            res_2 = CV_leaveoneout(DS_2, zscore_data=True)
            
            return pd.concat([res_1, res_2])
            
        else: # pure vanilla crossval
            
            res = CV_leaveoneout(DS, zscore_data=True)
            return res
    
# -------------------------------------------------------------------------------------------------

def decode_SplitHalf(DS, opt):
    
    task = opt.task
    model = opt.model
    
    if isExpUnexp(opt):
        
        expDS = DS[DS.sa.expected==1]
        unexpDS = DS[DS.sa.expected==0]
        
        if DivideInThirds(opt):
            
            if isWideNarrow(opt):
                
                res_exp = decode_thirds(expDS, opt, 'splithalf')
                res_unexp = SplitHalfCorr(unexpDS)
                return (res_exp, res_unexp)
            
            else: # test model 9
                
                res = decode_thirds((expDS, unexpDS), opt, 'splithalf')
                return res
        
        #else: (exp/unexp without dividing in thirds)
        
    else: # vanilla crossval, no expected vs. unexpected
        
        res = SplitHalfCorr(DS)
        return res   

# -------------------------------------------------------------------------------------------------

def distance_TrainTest(trainDS, testDS):
    '''
    TestDS is one third and it's either expected or unexpected.
    ---------------
    Computes distance (1 - corr) between A and B in a viewpoint-specific way
    (A30 vs. B30, A90 vs. B90)
    This way this distance can be compared between expected and unexpected.
    Idea: distance between A and B should be higher in expected.
    '''
    
    A30_test = np.mean(testDS[np.core.defchararray.find(testDS.sa.targets, 'A30')!=-1], axis=0)
    A30_train = np.mean(trainDS[np.core.defchararray.find(trainDS.sa.targets, 'A30')!=-1], axis=0)
    A90_test = np.mean(testDS[np.core.defchararray.find(testDS.sa.targets, 'A90')!=-1], axis=0)
    A90_train = np.mean(trainDS[np.core.defchararray.find(trainDS.sa.targets,'A90')!=-1], axis=0)
    # --------------------------------------------------------------------------------------
    B30_test = np.mean(testDS[np.core.defchararray.find(testDS.sa.targets, 'B30')!=-1], axis=0)
    B30_train = np.mean(trainDS[np.core.defchararray.find(trainDS.sa.targets, 'B30')!=-1], axis=0)
    B90_test = np.mean(testDS[np.core.defchararray.find(testDS.sa.targets, 'B90')!=-1], axis=0)
    B90_train = np.mean(trainDS[np.core.defchararray.find(trainDS.sa.targets,'B90')!=-1], axis=0)
    
    return 1-np.mean([pearsonr(A30_test, B30_train), pearsonr(A90_test, B90_train),
                    pearsonr(B30_test, A30_train), pearsonr(B90_test, A90_train)])
 