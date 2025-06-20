import pandas as pd
from nipype.interfaces.base import Bunch
import random
import re

# ---------------------------------------------------------------------------------

def specify_model_funcloc(eventsfile, model):
    '''
    Functional localizer:
    1 - faces, objects, scenes, scrambled, buttonpress
    2 - stimulus, baseline, buttonpress
    3 - object or scrambled, baseline, buttonpress
    '''
    
    events = pd.read_csv(eventsfile, sep='\t')
    
    onsets = []
    durations = []
    
    if model==1:
        
        conditions = ['faces', 'objects', 'scenes', 'scrambled', 'buttonpress']
    
        for r in conditions:
            onsets.append(list(events.loc[events['trial_type'] == r]['onset']))
            durations.append(list(events.loc[events['trial_type'] == r]['duration']))

    elif model==2:
        
        conditions = ['stimulus', 'baseline', 'buttonpress']
        
        # Stimulus:
        onsets.append(list(events.loc[events['trial_type'].isin(['faces', 'objects', 'scenes', 'scrambled'])]['onset']))
        durations.append(list(events.loc[events['trial_type'].isin(['faces', 'objects', 'scenes', 'scrambled'])]['duration']))
        
        # Baseline:
        onsets.append(list(events.loc[events['trial_type']=='blank']['onset']))
        durations.append(list(events.loc[events['trial_type']=='blank']['duration']))
        
        # Buttonpress:
        onsets.append(list(events.loc[events['trial_type']=='buttonpress']['onset']))
        durations.append(list(events.loc[events['trial_type']=='buttonpress']['duration']))
        
    elif model==3:
        
        conditions = ['objscr', 'baseline', 'buttonpress']
        
        # Stimulus:
        onsets.append(list(events.loc[events['trial_type'].isin(['objects', 'scrambled'])]['onset']))
        durations.append(list(events.loc[events['trial_type'].isin(['objects', 'scrambled'])]['duration']))
        
        # Baseline:
        onsets.append(list(events.loc[events['trial_type']=='blank']['onset']))
        durations.append(list(events.loc[events['trial_type']=='blank']['duration']))
        
        # Buttonpress:
        onsets.append(list(events.loc[events['trial_type']=='buttonpress']['onset']))
        durations.append(list(events.loc[events['trial_type']=='buttonpress']['duration']))
    
    else:
        raise Exception('"{:d}" is not a known model.'.format(model))
        
    evs = Bunch(conditions=conditions, onsets=onsets, durations=durations)
    
    return evs

# ---------------------------------------------------------------------------------

def specify_model_train(eventsfile, model):
    '''
    Models:
    1 - near (30°), far (90°), buttonpress
    2 - wide, narrow, buttonpress
    3 - wide, narrow, miniblock, buttonpress
    '''
    
    events = pd.read_csv(eventsfile, sep='\t')
    
    onsets = []
    durations = []
    
    if model==1:
        
        conditions = ['near', 'far']
        # ------------------- Near: -------------------
        onsets.append(list(events.loc[(events['rotation'] == 30) & (events['trial_type'] == 'miniblock')]['onset']))
        durations.append(list(events.loc[(events['rotation'] == 30) & (events['trial_type'] == 'miniblock')]['duration']))
        # ------------------- Far: -------------------
        onsets.append(list(events.loc[(events['rotation'] == 90) & (events['trial_type'] == 'miniblock')]['onset']))
        durations.append(list(events.loc[(events['rotation'] == 90) & (events['trial_type'] == 'miniblock')]['duration']))
        
    elif model==2:
        
        conditions = ['wide', 'narrow']
        wideindx = (((events['view']=='A') & (events['rotation']==30)) | ((events['view']=='B') & (events['rotation']==90))) & (events['trial_type']=='miniblock')
        narrindx = (((events['view']=='A') & (events['rotation']==90)) | ((events['view']=='B') & (events['rotation']==30))) & (events['trial_type']=='miniblock')
        # ------------------- Wide: -------------------
        onsets.append(list(events.loc[wideindx]['onset']))
        durations.append(list(events.loc[wideindx]['duration']))
        # ------------------- Narrow: -------------------
        onsets.append(list(events.loc[narrindx]['onset']))
        durations.append(list(events.loc[narrindx]['duration']))
    
    elif model==3:
        
        wideindx = ((events['view']=='A') & (events['rotation']==30)) | ((events['view']=='B') & (events['rotation']==90))
        narrindx = ((events['view']=='A') & (events['rotation']==90)) | ((events['view']=='B') & (events['rotation']==30))
        events_w = events.loc[(events['trial_type']=='miniblock') & (wideindx)]
        events_n = events.loc[(events['trial_type']=='miniblock') & (narrindx)]
        
        n_miniblocks = 10
        conditions = []
        for mb in range(1, n_miniblocks+1):
                conditions.append('wide_{:d}'.format(mb))
                onsets.append([events_w.iloc[mb-1]['onset']])
                durations.append([events_w.iloc[mb-1]['duration']])
        for mb in range(1, n_miniblocks+1):
                conditions.append('narrow_{:d}'.format(mb))
                onsets.append([events_n.iloc[mb-1]['onset']])
                durations.append([events_n.iloc[mb-1]['duration']])
                
    elif model==4: # also model 5
        
        A30indx = (events['view']=='A') & (events['rotation']==30)
        A90indx = (events['view']=='A') & (events['rotation']==90)
        B30indx = (events['view']=='B') & (events['rotation']==30)
        B90indx = (events['view']=='B') & (events['rotation']==90)
        events_A30 = events.loc[(events['trial_type']=='miniblock') & (A30indx)]
        events_A90 = events.loc[(events['trial_type']=='miniblock') & (A90indx)]
        events_B30 = events.loc[(events['trial_type']=='miniblock') & (B30indx)]
        events_B90 = events.loc[(events['trial_type']=='miniblock') & (B90indx)]
        
        n_miniblocks = 5
        conditions = []
        for mb in range(1, n_miniblocks+1):
            conditions.append('A30_{:d}'.format(mb))
            onsets.append([events_A30.iloc[mb-1]['onset']])
            durations.append([events_A30.iloc[mb-1]['duration']])
            # -----------------------
            conditions.append('A90_{:d}'.format(mb))
            onsets.append([events_A90.iloc[mb-1]['onset']])
            durations.append([events_A90.iloc[mb-1]['duration']])
            # -----------------------
            conditions.append('B30_{:d}'.format(mb))
            onsets.append([events_B30.iloc[mb-1]['onset']])
            durations.append([events_B30.iloc[mb-1]['duration']])
            # -----------------------
            conditions.append('B90_{:d}'.format(mb))
            onsets.append([events_B90.iloc[mb-1]['onset']])
            durations.append([events_B90.iloc[mb-1]['duration']])
            
    elif model==6:
        
        events_near = events.loc[(events['trial_type']=='miniblock') & (events['rotation']==30)]
        events_far = events.loc[(events['trial_type']=='miniblock') & (events['rotation']==90)]
        
        n_miniblocks = 10
        conditions = []
        for mb in range(1, n_miniblocks+1):
                conditions.append('near_{:d}'.format(mb))
                onsets.append([events_near.iloc[mb-1]['onset']])
                durations.append([events_near.iloc[mb-1]['duration']])
        for mb in range(1, n_miniblocks+1):
                conditions.append('far_{:d}'.format(mb))
                onsets.append([events_far.iloc[mb-1]['onset']])
                durations.append([events_far.iloc[mb-1]['duration']])
                
    elif model==7:
        
        events_A = events.loc[(events['trial_type']=='miniblock') & (events['view']=='A')]
        events_B = events.loc[(events['trial_type']=='miniblock') & (events['view']=='B')]
        
        n_miniblocks = 10
        conditions = []
        for mb in range(1, n_miniblocks+1):
                conditions.append('A_{:d}'.format(mb))
                onsets.append([events_A.iloc[mb-1]['onset']])
                durations.append([events_A.iloc[mb-1]['duration']])
        for mb in range(1, n_miniblocks+1):
                conditions.append('B_{:d}'.format(mb))
                onsets.append([events_B.iloc[mb-1]['onset']])
                durations.append([events_B.iloc[mb-1]['duration']])
        
    elif model==8:
        
        A90indx = (events['view']=='A') & (events['rotation']==90)
        B90indx = (events['view']=='B') & (events['rotation']==90)
        events_A90 = events.loc[(events['trial_type']=='miniblock') & (A90indx)]
        events_B90 = events.loc[(events['trial_type']=='miniblock') & (B90indx)]
        
        n_miniblocks = 5
        conditions = []
        for mb in range(1, n_miniblocks+1):
            conditions.append('A90_{:d}'.format(mb))
            onsets.append([events_A90.iloc[mb-1]['onset']])
            durations.append([events_A90.iloc[mb-1]['duration']])
            # -----------------------
            conditions.append('B90_{:d}'.format(mb))
            onsets.append([events_B90.iloc[mb-1]['onset']])
            durations.append([events_B90.iloc[mb-1]['duration']])
    else:
        raise Exception('"{:d}" is not a known model.'.format(model))
    
    if events['trial_type'].str.contains('buttonpress').any(): # if there's at least one response
        conditions.append('buttonpress')
        onsets.append(list(events.loc[events['trial_type'] == 'buttonpress']['onset']))
        durations.append(list(events.loc[events['trial_type'] == 'buttonpress']['duration']))
    
    evs = Bunch(conditions=conditions, onsets=onsets, durations=durations)
        
    return evs

# ---------------------------------------------------------------------------------

def specify_model_test(eventsfile, model, behav):
    """
    Events:
    1 - Fixation (0.5 - 1.5 sec depending on previous trial's response time)
    2 - Scene view 1 (2 sec)
    3 - Scene view 2 (0.5 sec)
    4 - Scene view 3 (0.5 sec) **
    5 - Scene view 4 (0.5 sec) **
    6 - (Final) view 5 (1-1.5 sec) **
    7 - Probe 1 (0.05 sec)
    8 - ISI (0.1 sec)
    9 - Probe 2 (0.05 sec)
    10 - Probe offset/pre-response delay (0.05 sec)
    11 - Response window starts (until response, max 1.5 sec)
    12 - Response
    
    ** = occluder
    
    """
    
    events = pd.read_csv(eventsfile, sep='\t')
    
    onsets = []
    durations = []

    if model==1:
        # events: from 2 to 4 (scene onset to mask onset, when object was physically present)
        conditions = ['bed', 'couch']
        bedscenes = [2, 5, 6, 7, 9, 13, 14, 17, 18, 20]
        bedindx = behav.index[behav['Scene'].isin(bedscenes)]
        bedtrials = events[events['trial_no'].isin(bedindx)]
        couchtrials = events[~events['trial_no'].isin(bedindx)]
        onsets.append(list(bedtrials[bedtrials['event_no']==2].onset))
        onsets.append(list(couchtrials[couchtrials['event_no']==2].onset))
        durations.append(list(bedtrials[bedtrials['event_no']==4].onset.values - bedtrials[bedtrials['event_no']==2].onset.values))
        durations.append(list(couchtrials[couchtrials['event_no']==4].onset.values - couchtrials[couchtrials['event_no']==2].onset.values))

    elif model==2:
        # events: 6 to 10 (last scene onset-offset)
        conditions = ['near', 'far']
        nearindx = behav.index[behav['FinalView']==30]
        neartrials = events[events['trial_no'].isin(nearindx)]
        fartrials = events[~events['trial_no'].isin(nearindx)]
        onsets.append(list(neartrials[neartrials['event_no']==6].onset))
        onsets.append(list(fartrials[fartrials['event_no']==6].onset))
        durations.append(list(neartrials[neartrials['event_no']==10].onset.values - neartrials[neartrials['event_no']==6].onset.values))
        durations.append(list(fartrials[fartrials['event_no']==10].onset.values - fartrials[fartrials['event_no']==6].onset.values))

    elif model==3:
        # events: from 2 to 3 (scene onset to first rotation)
        conditions = ['init_wide', 'init_narrow']
        wideindx = behav.index[behav['InitView']==1]
        widetrials = events[events['trial_no'].isin(wideindx)]
        narrtrials = events[~events['trial_no'].isin(wideindx)]
        onsets.append(list(widetrials[widetrials['event_no']==2].onset))
        onsets.append(list(narrtrials[narrtrials['event_no']==2].onset))
        durations.append(list(widetrials[widetrials['event_no']==3].onset.values - widetrials[widetrials['event_no']==2].onset.values))
        durations.append(list(narrtrials[narrtrials['event_no']==3].onset.values - narrtrials[narrtrials['event_no']==2].onset.values))

    elif model==4:
        # events: 7 (first probe)
        conditions = ['final_wide', 'final_narrow']
        
        wideindx = (((behav['InitView']==1) & (behav['FinalView']==30)) | ((behav['InitView']==2) & (behav['FinalView']==90)))
        narrindx = (((behav['InitView']==1) & (behav['FinalView']==90)) | ((behav['InitView']==2) & (behav['FinalView']==30)))
        widewhere = behav.index[(wideindx & (behav['Consistent']==1)) | (narrindx & (behav['Consistent']==0))]
        narrwhere = behav.index[(narrindx & (behav['Consistent']==1)) | (wideindx & (behav['Consistent']==0))]
        
        widetrials = events[events['trial_no'].isin(widewhere)]
        narrtrials = events[events['trial_no'].isin(narrwhere)]
        onsets.append(list(widetrials[widetrials['event_no']==7].onset))
        onsets.append(list(narrtrials[narrtrials['event_no']==7].onset))
        durations.append([0] * len(onsets[0]))
        durations.append([0] * len(onsets[1]))

    elif model==5:
        # events: 7 (first probe)
        conditions = ['expected', 'unexpected']
        expindx = behav.index[behav['Consistent']==1]
        exptrials = events[events['trial_no'].isin(expindx)]
        unexptrials = events[~events['trial_no'].isin(expindx)]
        onsets.append(list(exptrials[exptrials['event_no']==7].onset))
        onsets.append(list(unexptrials[unexptrials['event_no']==7].onset))
        durations.append([0] * len(onsets[0]))
        durations.append([0] * len(onsets[1]))

    elif model==6:
        # events: 7 (first probe)
        conditions = ['exp_wide', 'exp_narrow', 'unexp_wide', 'unexp_narrow']
        widemask = ((behav['InitView']==1) & (behav['FinalView']==30)) | ((behav['InitView']==2) & (behav['FinalView']==90))
        narrmask = ~widemask
        EW_indx = behav.index[(behav['Consistent']==1) & widemask] # expected, wide
        EN_indx = behav.index[(behav['Consistent']==1) & narrmask] # expected, narrow
        
        # the unexpected ones NEED TO BE SWAPPED:
        UW_indx = behav.index[(behav['Consistent']==0) & narrmask] # unexpected, wide
        UN_indx = behav.index[(behav['Consistent']==0) & widemask] # unexpected, narrow
        
        # these are WRONG but I keep them for consistency:
        #UW_indx = behav.index[(behav['Consistent']==0) & widemask] # unexpected, wide
        #UN_indx = behav.index[(behav['Consistent']==0) & narrmask] # unexpected, narrow
        
        EW_trials = events[events['trial_no'].isin(EW_indx)]
        EN_trials = events[events['trial_no'].isin(EN_indx)]
        UW_trials = events[events['trial_no'].isin(UW_indx)]
        UN_trials = events[events['trial_no'].isin(UN_indx)]
        
        onsets.append(list(EW_trials[EW_trials['event_no']==7].onset))
        onsets.append(list(EN_trials[EN_trials['event_no']==7].onset))
        onsets.append(list(UW_trials[UW_trials['event_no']==7].onset))
        onsets.append(list(UN_trials[UN_trials['event_no']==7].onset))
        for o in onsets:
            durations.append([0] * len(o))
    
    elif model==7: # control by randomly selecting 1/3 of expected trials
        # events: 7 (first probe)
        conditions = ['exp_wide_1', 'exp_wide_2', 'exp_wide_3',
                      'exp_narrow_1', 'exp_narrow_2', 'exp_narrow_3', 
                      'unexp_wide', 'unexp_narrow']
        widemask = ((behav['InitView']==1) & (behav['FinalView']==30)) | ((behav['InitView']==2) & (behav['FinalView']==90))
        narrmask = ~widemask
        EW_indx = behav.index[(behav['Consistent']==1) & widemask] # expected, wide
        EN_indx = behav.index[(behav['Consistent']==1) & narrmask] # expected, narrow
        
        # the unexpected ones NEED TO BE SWAPPED:
        UW_indx = behav.index[(behav['Consistent']==0) & narrmask] # unexpected, wide
        UN_indx = behav.index[(behav['Consistent']==0) & widemask] # unexpected, narrow
        
        # these are WRONG but I keep them for consistency:
        #UW_indx = behav.index[(behav['Consistent']==0) & widemask] # unexpected, wide
        #UN_indx = behav.index[(behav['Consistent']==0) & narrmask] # unexpected, narrow
        
        # ---------------------------------
        EW_trials = events[events['trial_no'].isin(EW_indx)]
        EW_trialnos = list(EW_trials.trial_no.unique())
        EW_trialnos = random.sample(EW_trialnos, len(EW_trialnos))
        EW_trials_1 = events[events['trial_no'].isin(EW_trialnos[:6])]
        EW_trials_2 = events[events['trial_no'].isin(EW_trialnos[6:12])]
        EW_trials_3 = events[events['trial_no'].isin(EW_trialnos[12:18])]
        # ---------------------------------
        EN_trials = events[events['trial_no'].isin(EN_indx)]
        EN_trialnos = list(EN_trials.trial_no.unique())
        EN_trialnos = random.sample(EN_trialnos, len(EN_trialnos))
        EN_trials_1 = events[events['trial_no'].isin(EN_trialnos[:6])]
        EN_trials_2 = events[events['trial_no'].isin(EN_trialnos[6:12])]
        EN_trials_3 = events[events['trial_no'].isin(EN_trialnos[12:18])]
        
        UW_trials = events[events['trial_no'].isin(UW_indx)]
        UN_trials = events[events['trial_no'].isin(UN_indx)]
        
        onsets.append(list(EW_trials_1[EW_trials_1['event_no']==7].onset))
        onsets.append(list(EW_trials_2[EW_trials_2['event_no']==7].onset))
        onsets.append(list(EW_trials_3[EW_trials_3['event_no']==7].onset))
        onsets.append(list(EN_trials_1[EN_trials_1['event_no']==7].onset))
        onsets.append(list(EN_trials_2[EN_trials_2['event_no']==7].onset))
        onsets.append(list(EN_trials_3[EN_trials_3['event_no']==7].onset))
        
        onsets.append(list(UW_trials[UW_trials['event_no']==7].onset))
        onsets.append(list(UN_trials[UN_trials['event_no']==7].onset))
        for o in onsets:
            durations.append([0] * len(o))
    
    elif model==8:
        
        # events: 7 (first probe)
        conditions = ['exp_wide', 'exp_narrow', 'unexp_wide', 'unexp_narrow']
        widemask = ((behav['InitView']==1) & (behav['FinalView']==30)) | ((behav['InitView']==2) & (behav['FinalView']==90))
        narrmask = ~widemask
        widemask[behav['Consistent']==0] = ~widemask[behav['Consistent']==0] # swap w/n for unexpected
        narrmask[behav['Consistent']==0] = ~narrmask[behav['Consistent']==0]
        W_indx = behav.index[widemask]
        N_indx = behav.index[narrmask]
        
        W_trialnos = list(W_indx)
        W_trialnos = random.sample(W_trialnos, len(W_trialnos))
        EW_trialnos = W_trialnos[:126] # randomly assign 75% to "expected" condition
        UW_trialnos = W_trialnos[126:] # and the rest to "unexpected"

        N_trialnos = list(N_indx)
        N_trialnos = random.sample(N_trialnos, len(N_trialnos))
        EN_trialnos = N_trialnos[:126] # randomly assign 75% to "expected" condition
        UN_trialnos = N_trialnos[126:] # and the rest to "unexpected"
        
        EW_trials = events[events['trial_no'].isin(EW_trialnos)]
        EN_trials = events[events['trial_no'].isin(EN_trialnos)]
        UW_trials = events[events['trial_no'].isin(UW_trialnos)]
        UN_trials = events[events['trial_no'].isin(UN_trialnos)]
        
        onsets.append(list(EW_trials[EW_trials['event_no']==7].onset))
        onsets.append(list(EN_trials[EN_trials['event_no']==7].onset))
        onsets.append(list(UW_trials[UW_trials['event_no']==7].onset))
        onsets.append(list(UN_trials[UN_trials['event_no']==7].onset))
        for o in onsets:
            durations.append([0] * len(o))
    
    elif model==9:
        
        # events: 7 (first probe)
        conditions = ['expected_1', 'expected_2', 'expected_3', 'unexpected']
        expindx = behav.index[behav['Consistent']==1]
        exptrials = events[events['trial_no'].isin(expindx)]
        unexptrials = events[~events['trial_no'].isin(expindx)]
        
        exp_trialnos = list(exptrials.trial_no.unique())
        exp_trialnos = random.sample(exp_trialnos, len(exp_trialnos))
        exp_trials_1 = events[events['trial_no'].isin(exp_trialnos[:12])]
        exp_trials_2 = events[events['trial_no'].isin(exp_trialnos[12:24])]
        exp_trials_3 = events[events['trial_no'].isin(exp_trialnos[24:36])]
        
        onsets.append(list(exp_trials_1[exp_trials_1['event_no']==7].onset))
        onsets.append(list(exp_trials_2[exp_trials_2['event_no']==7].onset))
        onsets.append(list(exp_trials_3[exp_trials_3['event_no']==7].onset))
        onsets.append(list(unexptrials[unexptrials['event_no']==7].onset))
        for o in onsets:
            durations.append([0] * len(o))
    
    elif model==10:
        
        # events: 6 - 10 (from final scene view onset to second probe offset)
        conditions = ['final_wide', 'final_narrow']
        wideindx = behav.index[((behav['InitView']==1) & (behav['FinalView']==30)) | ((behav['InitView']==2) & (behav['FinalView']==90))]
        widetrials = events[events['trial_no'].isin(wideindx)]
        narrtrials = events[~events['trial_no'].isin(wideindx)]
        onsets.append(list(widetrials[widetrials['event_no']==6].onset))
        onsets.append(list(narrtrials[narrtrials['event_no']==6].onset))
        durations.append(list(widetrials[widetrials['event_no']==10].onset.values - widetrials[widetrials['event_no']==6].onset.values))
        durations.append(list(narrtrials[narrtrials['event_no']==10].onset.values - narrtrials[narrtrials['event_no']==6].onset.values))
    
    elif model==11:
        
        # events: 6 - 10 (from final scene view onset to second probe offset)
        conditions = ['exp_wide_1', 'exp_wide_2', 'exp_wide_3',
                      'exp_narrow_1', 'exp_narrow_2', 'exp_narrow_3', 
                      'unexp_wide', 'unexp_narrow']
        widemask = ((behav['InitView']==1) & (behav['FinalView']==30)) | ((behav['InitView']==2) & (behav['FinalView']==90))
        narrmask = ~widemask
        EW_indx = behav.index[(behav['Consistent']==1) & widemask] # expected, wide
        EN_indx = behav.index[(behav['Consistent']==1) & narrmask] # expected, narrow
        
        # the unexpected ones NEED TO BE SWAPPED:
        UW_indx = behav.index[(behav['Consistent']==0) & narrmask] # unexpected, wide
        UN_indx = behav.index[(behav['Consistent']==0) & widemask] # unexpected, narrow
        
        # these are WRONG but I keep them for consistency:
        #UW_indx = behav.index[(behav['Consistent']==0) & widemask] # unexpected, wide
        #UN_indx = behav.index[(behav['Consistent']==0) & narrmask] # unexpected, narrow
        
        # ---------------------------------
        EW_trials = events[events['trial_no'].isin(EW_indx)]
        EW_trialnos = list(EW_trials.trial_no.unique())
        EW_trialnos = random.sample(EW_trialnos, len(EW_trialnos))
        EW_trials_1 = events[events['trial_no'].isin(EW_trialnos[:6])]
        EW_trials_2 = events[events['trial_no'].isin(EW_trialnos[6:12])]
        EW_trials_3 = events[events['trial_no'].isin(EW_trialnos[12:18])]
        # ---------------------------------
        EN_trials = events[events['trial_no'].isin(EN_indx)]
        EN_trialnos = list(EN_trials.trial_no.unique())
        EN_trialnos = random.sample(EN_trialnos, len(EN_trialnos))
        EN_trials_1 = events[events['trial_no'].isin(EN_trialnos[:6])]
        EN_trials_2 = events[events['trial_no'].isin(EN_trialnos[6:12])]
        EN_trials_3 = events[events['trial_no'].isin(EN_trialnos[12:18])]
        
        UW_trials = events[events['trial_no'].isin(UW_indx)]
        UN_trials = events[events['trial_no'].isin(UN_indx)]
        
        onsets.append(list(EW_trials_1[EW_trials_1['event_no']==6].onset))
        onsets.append(list(EW_trials_2[EW_trials_2['event_no']==6].onset))
        onsets.append(list(EW_trials_3[EW_trials_3['event_no']==6].onset))
        onsets.append(list(EN_trials_1[EN_trials_1['event_no']==6].onset))
        onsets.append(list(EN_trials_2[EN_trials_2['event_no']==6].onset))
        onsets.append(list(EN_trials_3[EN_trials_3['event_no']==6].onset))
        
        onsets.append(list(UW_trials[UW_trials['event_no']==6].onset))
        onsets.append(list(UN_trials[UN_trials['event_no']==6].onset))
        
        durations.append(list(EW_trials_1[EW_trials_1['event_no']==10].onset.values - EW_trials_1[EW_trials_1['event_no']==6].onset.values))
        durations.append(list(EW_trials_2[EW_trials_2['event_no']==10].onset.values - EW_trials_2[EW_trials_2['event_no']==6].onset.values))
        durations.append(list(EW_trials_3[EW_trials_3['event_no']==10].onset.values - EW_trials_3[EW_trials_3['event_no']==6].onset.values))
        durations.append(list(EN_trials_1[EN_trials_1['event_no']==10].onset.values - EN_trials_1[EN_trials_1['event_no']==6].onset.values))
        durations.append(list(EN_trials_2[EN_trials_2['event_no']==10].onset.values - EN_trials_2[EN_trials_2['event_no']==6].onset.values))
        durations.append(list(EN_trials_3[EN_trials_3['event_no']==10].onset.values - EN_trials_3[EN_trials_3['event_no']==6].onset.values))
        
        durations.append(list(UW_trials[UW_trials['event_no']==10].onset.values - UW_trials[UW_trials['event_no']==6].onset.values))
        durations.append(list(UN_trials[UN_trials['event_no']==10].onset.values - UN_trials[UN_trials['event_no']==6].onset.values))
    
    elif model==12:
        
        # events: 7 (first probe)
        conditions = ['A_30_exp_1', 'A_30_exp_2', 'A_30_exp_3', 'A_30_unexp',
                      'A_90_exp_1', 'A_90_exp_2', 'A_90_exp_3', 'A_90_unexp', 
                      'B_30_exp_1', 'B_30_exp_2', 'B_30_exp_3', 'B_30_unexp',
                      'B_90_exp_1', 'B_90_exp_2', 'B_90_exp_3', 'B_90_unexp']
        A30mask = ((behav['InitView']==1) & (behav['FinalView']==30))
        A90mask = ((behav['InitView']==1) & (behav['FinalView']==90))
        B30mask = ((behav['InitView']==2) & (behav['FinalView']==30))
        B90mask = ((behav['InitView']==2) & (behav['FinalView']==90))
        
        A30_E_indx = behav.index[(behav['Consistent']==1) & A30mask]
        A90_E_indx = behav.index[(behav['Consistent']==1) & A90mask]
        B30_E_indx = behav.index[(behav['Consistent']==1) & B30mask]
        B90_E_indx = behav.index[(behav['Consistent']==1) & B90mask]
        
        # the unexpected ones NEED TO BE SWAPPED:
        # (unexpected stimulus came from the other initial viewpoint)
        A30_U_indx = behav.index[(behav['Consistent']==0) & B30mask]
        A90_U_indx = behav.index[(behav['Consistent']==0) & B90mask]
        B30_U_indx = behav.index[(behav['Consistent']==0) & A30mask]
        B90_U_indx = behav.index[(behav['Consistent']==0) & A90mask]
        
        # ---------------------------------
        A30_E_trials = events[events['trial_no'].isin(A30_E_indx)]
        A30_E_trialnos = list(A30_E_trials.trial_no.unique())
        A30_E_trialnos = random.sample(A30_E_trialnos, len(A30_E_trialnos))
        A30_E_trials_1 = events[events['trial_no'].isin(A30_E_trialnos[:3])]
        A30_E_trials_2 = events[events['trial_no'].isin(A30_E_trialnos[3:6])]
        A30_E_trials_3 = events[events['trial_no'].isin(A30_E_trialnos[6:9])]
        # ---------------------------------
        A90_E_trials = events[events['trial_no'].isin(A90_E_indx)]
        A90_E_trialnos = list(A90_E_trials.trial_no.unique())
        A90_E_trialnos = random.sample(A90_E_trialnos, len(A90_E_trialnos))
        A90_E_trials_1 = events[events['trial_no'].isin(A90_E_trialnos[:3])]
        A90_E_trials_2 = events[events['trial_no'].isin(A90_E_trialnos[3:6])]
        A90_E_trials_3 = events[events['trial_no'].isin(A90_E_trialnos[6:9])]
        # ---------------------------------
        B30_E_trials = events[events['trial_no'].isin(B30_E_indx)]
        B30_E_trialnos = list(B30_E_trials.trial_no.unique())
        B30_E_trialnos = random.sample(B30_E_trialnos, len(B30_E_trialnos))
        B30_E_trials_1 = events[events['trial_no'].isin(B30_E_trialnos[:3])]
        B30_E_trials_2 = events[events['trial_no'].isin(B30_E_trialnos[3:6])]
        B30_E_trials_3 = events[events['trial_no'].isin(B30_E_trialnos[6:9])]
        # ---------------------------------
        B90_E_trials = events[events['trial_no'].isin(B90_E_indx)]
        B90_E_trialnos = list(B90_E_trials.trial_no.unique())
        B90_E_trialnos = random.sample(B90_E_trialnos, len(B90_E_trialnos))
        B90_E_trials_1 = events[events['trial_no'].isin(B90_E_trialnos[:3])]
        B90_E_trials_2 = events[events['trial_no'].isin(B90_E_trialnos[3:6])]
        B90_E_trials_3 = events[events['trial_no'].isin(B90_E_trialnos[6:9])]
        # ---------------------------------
        
        A30_U_trials = events[events['trial_no'].isin(A30_U_indx)]
        A90_U_trials = events[events['trial_no'].isin(A90_U_indx)]
        B30_U_trials = events[events['trial_no'].isin(B30_U_indx)]
        B90_U_trials = events[events['trial_no'].isin(B90_U_indx)]
        
        # Add to onsets and durations:
        
        onsets.append(list(A30_E_trials_1[A30_E_trials_1['event_no']==7].onset))
        onsets.append(list(A30_E_trials_2[A30_E_trials_2['event_no']==7].onset))
        onsets.append(list(A30_E_trials_3[A30_E_trials_3['event_no']==7].onset))
        
        onsets.append(list(A30_U_trials[A30_U_trials['event_no']==7].onset))
        
        onsets.append(list(A90_E_trials_1[A90_E_trials_1['event_no']==7].onset))
        onsets.append(list(A90_E_trials_2[A90_E_trials_2['event_no']==7].onset))
        onsets.append(list(A90_E_trials_3[A90_E_trials_3['event_no']==7].onset))
        
        onsets.append(list(A90_U_trials[A90_U_trials['event_no']==7].onset))
        
        onsets.append(list(B30_E_trials_1[B30_E_trials_1['event_no']==7].onset))
        onsets.append(list(B30_E_trials_2[B30_E_trials_2['event_no']==7].onset))
        onsets.append(list(B30_E_trials_3[B30_E_trials_3['event_no']==7].onset))
        
        onsets.append(list(B30_U_trials[B30_U_trials['event_no']==7].onset))
        
        onsets.append(list(B90_E_trials_1[B90_E_trials_1['event_no']==7].onset))
        onsets.append(list(B90_E_trials_2[B90_E_trials_2['event_no']==7].onset))
        onsets.append(list(B90_E_trials_3[B90_E_trials_3['event_no']==7].onset))
        
        onsets.append(list(B90_U_trials[B90_U_trials['event_no']==7].onset))
        
        conditions = [c for i, c in enumerate(conditions) if onsets[i]!=[]]
        onsets = [o for o in onsets if o != []]
        
        for o in onsets:
            durations.append([0] * len(o))
    
    elif model==13:
        
        # events: 7 (first probe)
        conditions = ['rot30_exp_1', 'rot30_exp_2', 'rot30_exp_3', 'rot30_unexp',
                      'rot90_exp_1', 'rot90_exp_2', 'rot90_exp_3', 'rot90_unexp']
        E30_indx = behav.index[(behav['Consistent']==1) & (behav['FinalView']==30)] # expected, 30°
        E90_indx = behav.index[(behav['Consistent']==1) & (behav['FinalView']==90)] # expected, 90°
        
        U30_indx = behav.index[(behav['Consistent']==0) & (behav['FinalView']==30)] # unexpected, 30°
        U90_indx = behav.index[(behav['Consistent']==0) & (behav['FinalView']==90)] # unexpected, 90°
        
        # ---------------------------------
        E30_trials = events[events['trial_no'].isin(E30_indx)]
        E30_trialnos = list(E30_trials.trial_no.unique())
        E30_trialnos = random.sample(E30_trialnos, len(E30_trialnos))
        E30_trials_1 = events[events['trial_no'].isin(E30_trialnos[:6])]
        E30_trials_2 = events[events['trial_no'].isin(E30_trialnos[6:12])]
        E30_trials_3 = events[events['trial_no'].isin(E30_trialnos[12:18])]
        # ---------------------------------
        E90_trials = events[events['trial_no'].isin(E90_indx)]
        E90_trialnos = list(E90_trials.trial_no.unique())
        E90_trialnos = random.sample(E90_trialnos, len(E90_trialnos))
        E90_trials_1 = events[events['trial_no'].isin(E90_trialnos[:6])]
        E90_trials_2 = events[events['trial_no'].isin(E90_trialnos[6:12])]
        E90_trials_3 = events[events['trial_no'].isin(E90_trialnos[12:18])]
        
        U30_trials = events[events['trial_no'].isin(U30_indx)]
        U90_trials = events[events['trial_no'].isin(U90_indx)]
        
        onsets.append(list(E30_trials_1[E30_trials_1['event_no']==7].onset))
        onsets.append(list(E30_trials_2[E30_trials_2['event_no']==7].onset))
        onsets.append(list(E30_trials_3[E30_trials_3['event_no']==7].onset))
        
        onsets.append(list(U30_trials[U30_trials['event_no']==7].onset))
        
        onsets.append(list(E90_trials_1[E90_trials_1['event_no']==7].onset))
        onsets.append(list(E90_trials_2[E90_trials_2['event_no']==7].onset))
        onsets.append(list(E90_trials_3[E90_trials_3['event_no']==7].onset))
        
        onsets.append(list(U90_trials[U90_trials['event_no']==7].onset))
        for o in onsets:
            durations.append([0] * len(o))
    
    elif model==14:
        # events: 7 (first probe)
        conditions = ['rot30_exp', 'rot30_unexp',
                      'rot90_exp', 'rot90_unexp']
        
        E30_indx = behav.index[(behav['Consistent']==1) & (behav['FinalView']==30)] # expected, 30°
        E90_indx = behav.index[(behav['Consistent']==1) & (behav['FinalView']==90)] # expected, 90°
        U30_indx = behav.index[(behav['Consistent']==0) & (behav['FinalView']==30)] # unexpected, 30°
        U90_indx = behav.index[(behav['Consistent']==0) & (behav['FinalView']==90)] # unexpected, 90°
        
        # ---------------------------------------------------
        E30_trials = events[events['trial_no'].isin(E30_indx)]
        E90_trials = events[events['trial_no'].isin(E90_indx)]
        U30_trials = events[events['trial_no'].isin(U30_indx)]
        U90_trials = events[events['trial_no'].isin(U90_indx)]
        
        onsets.append(list(E30_trials[E30_trials['event_no']==7].onset))
        onsets.append(list(U30_trials[U30_trials['event_no']==7].onset))
        onsets.append(list(E90_trials[E90_trials['event_no']==7].onset))
        onsets.append(list(U90_trials[U90_trials['event_no']==7].onset))
        for o in onsets:
            durations.append([0] * len(o))
    
    elif model==16:
        # events: 7 (first probe)
        conditions = ['A30', 'A90', 'B30', 'B90']
        
        A30mask = ((behav['InitView']==1) & (behav['FinalView']==30))
        A90mask = ((behav['InitView']==1) & (behav['FinalView']==90))
        B30mask = ((behav['InitView']==2) & (behav['FinalView']==30))
        B90mask = ((behav['InitView']==2) & (behav['FinalView']==90))
        
        A30_indx = behav.index[((behav['Consistent']==1) & A30mask) | ((behav['Consistent']==0) & B30mask)]
        A90_indx = behav.index[((behav['Consistent']==1) & A90mask) | ((behav['Consistent']==0) & B90mask)]
        B30_indx = behav.index[((behav['Consistent']==1) & B30mask) | ((behav['Consistent']==0) & A30mask)]
        B90_indx = behav.index[((behav['Consistent']==1) & B90mask) | ((behav['Consistent']==0) & A90mask)]
        
        A30_trials = events[events['trial_no'].isin(A30_indx)]
        A90_trials = events[events['trial_no'].isin(A90_indx)]
        B30_trials = events[events['trial_no'].isin(B30_indx)]
        B90_trials = events[events['trial_no'].isin(B90_indx)]
        
        onsets.append(list(A30_trials[A30_trials['event_no']==7].onset))
        onsets.append(list(A90_trials[A90_trials['event_no']==7].onset))
        onsets.append(list(B30_trials[B30_trials['event_no']==7].onset))
        onsets.append(list(B90_trials[B90_trials['event_no']==7].onset))
        for o in onsets:
            durations.append([0] * len(o))
            
    elif model==18:
        # events: 7 (first probe)
        conditions = ['A', 'B']
        A_indx = behav.index[behav['InitView']==1]
        B_indx = behav.index[behav['InitView']==2]
        A_trials = events[events['trial_no'].isin(A_indx)]
        B_trials = events[events['trial_no'].isin(B_indx)]
        
        onsets.append(list(A_trials[A_trials['event_no']==7].onset))
        onsets.append(list(B_trials[B_trials['event_no']==7].onset))
        for o in onsets:
            durations.append([0] * len(o))
            
    elif model==19:
        # events: 7 (first probe)
        conditions = ['A_exp_1', 'A_exp_2', 'A_exp_3', 'A_unexp',
                      'B_exp_1', 'B_exp_2', 'B_exp_3', 'B_unexp']
        EA_indx = behav.index[(behav['Consistent']==1) & (behav['InitView']==1)] # expected, A
        EB_indx = behav.index[(behav['Consistent']==1) & (behav['InitView']==2)] # expected, B
        UA_indx = behav.index[(behav['Consistent']==0) & (behav['InitView']==2)] # unexpected, A
        UB_indx = behav.index[(behav['Consistent']==0) & (behav['InitView']==1)] # unexpected, B
        
        # ---------------------------------
        EA_trials = events[events['trial_no'].isin(EA_indx)]
        EA_trialnos = list(EA_trials.trial_no.unique())
        EA_trialnos = random.sample(EA_trialnos, len(EA_trialnos))
        EA_trials_1 = events[events['trial_no'].isin(EA_trialnos[:6])]
        EA_trials_2 = events[events['trial_no'].isin(EA_trialnos[6:12])]
        EA_trials_3 = events[events['trial_no'].isin(EA_trialnos[12:18])]
        
        UA_trials = events[events['trial_no'].isin(UA_indx)]
        
        # ---------------------------------
        EB_trials = events[events['trial_no'].isin(EB_indx)]
        EB_trialnos = list(EB_trials.trial_no.unique())
        EB_trialnos = random.sample(EB_trialnos, len(EB_trialnos))
        EB_trials_1 = events[events['trial_no'].isin(EB_trialnos[:6])]
        EB_trials_2 = events[events['trial_no'].isin(EB_trialnos[6:12])]
        EB_trials_3 = events[events['trial_no'].isin(EB_trialnos[12:18])]
        
        UB_trials = events[events['trial_no'].isin(UB_indx)]
        
        onsets.append(list(EA_trials_1[EA_trials_1['event_no']==7].onset))
        onsets.append(list(EA_trials_2[EA_trials_2['event_no']==7].onset))
        onsets.append(list(EA_trials_3[EA_trials_3['event_no']==7].onset))
        
        onsets.append(list(UA_trials[UA_trials['event_no']==7].onset))
        
        onsets.append(list(EB_trials_1[EB_trials_1['event_no']==7].onset))
        onsets.append(list(EB_trials_2[EB_trials_2['event_no']==7].onset))
        onsets.append(list(EB_trials_3[EB_trials_3['event_no']==7].onset))
        
        onsets.append(list(UB_trials[UB_trials['event_no']==7].onset))
        for o in onsets:
            durations.append([0] * len(o))
    
    elif model==20:
        # events: 6 - 10 (from final scene view onset to second probe offset)
        conditions = ['A_30_exp_1', 'A_30_exp_2', 'A_30_exp_3', 'A_30_unexp',
                      'A_90_exp_1', 'A_90_exp_2', 'A_90_exp_3', 'A_90_unexp', 
                      'B_30_exp_1', 'B_30_exp_2', 'B_30_exp_3', 'B_30_unexp',
                      'B_90_exp_1', 'B_90_exp_2', 'B_90_exp_3', 'B_90_unexp']
        A30mask = ((behav['InitView']==1) & (behav['FinalView']==30))
        A90mask = ((behav['InitView']==1) & (behav['FinalView']==90))
        B30mask = ((behav['InitView']==2) & (behav['FinalView']==30))
        B90mask = ((behav['InitView']==2) & (behav['FinalView']==90))
        
        A30_E_indx = behav.index[(behav['Consistent']==1) & A30mask]
        A90_E_indx = behav.index[(behav['Consistent']==1) & A90mask]
        B30_E_indx = behav.index[(behav['Consistent']==1) & B30mask]
        B90_E_indx = behav.index[(behav['Consistent']==1) & B90mask]
        
        # the unexpected ones NEED TO BE SWAPPED:
        # (unexpected stimulus came from the other initial viewpoint)
        A30_U_indx = behav.index[(behav['Consistent']==0) & B30mask]
        A90_U_indx = behav.index[(behav['Consistent']==0) & B90mask]
        B30_U_indx = behav.index[(behav['Consistent']==0) & A30mask]
        B90_U_indx = behav.index[(behav['Consistent']==0) & A90mask]
        
        # ---------------------------------
        A30_E_trials = events[events['trial_no'].isin(A30_E_indx)]
        A30_E_trialnos = list(A30_E_trials.trial_no.unique())
        A30_E_trialnos = random.sample(A30_E_trialnos, len(A30_E_trialnos))
        A30_E_trials_1 = events[events['trial_no'].isin(A30_E_trialnos[:3])]
        A30_E_trials_2 = events[events['trial_no'].isin(A30_E_trialnos[3:6])]
        A30_E_trials_3 = events[events['trial_no'].isin(A30_E_trialnos[6:9])]
        # ---------------------------------
        A90_E_trials = events[events['trial_no'].isin(A90_E_indx)]
        A90_E_trialnos = list(A90_E_trials.trial_no.unique())
        A90_E_trialnos = random.sample(A90_E_trialnos, len(A90_E_trialnos))
        A90_E_trials_1 = events[events['trial_no'].isin(A90_E_trialnos[:3])]
        A90_E_trials_2 = events[events['trial_no'].isin(A90_E_trialnos[3:6])]
        A90_E_trials_3 = events[events['trial_no'].isin(A90_E_trialnos[6:9])]
        # ---------------------------------
        B30_E_trials = events[events['trial_no'].isin(B30_E_indx)]
        B30_E_trialnos = list(B30_E_trials.trial_no.unique())
        B30_E_trialnos = random.sample(B30_E_trialnos, len(B30_E_trialnos))
        B30_E_trials_1 = events[events['trial_no'].isin(B30_E_trialnos[:3])]
        B30_E_trials_2 = events[events['trial_no'].isin(B30_E_trialnos[3:6])]
        B30_E_trials_3 = events[events['trial_no'].isin(B30_E_trialnos[6:9])]
        # ---------------------------------
        B90_E_trials = events[events['trial_no'].isin(B90_E_indx)]
        B90_E_trialnos = list(B90_E_trials.trial_no.unique())
        B90_E_trialnos = random.sample(B90_E_trialnos, len(B90_E_trialnos))
        B90_E_trials_1 = events[events['trial_no'].isin(B90_E_trialnos[:3])]
        B90_E_trials_2 = events[events['trial_no'].isin(B90_E_trialnos[3:6])]
        B90_E_trials_3 = events[events['trial_no'].isin(B90_E_trialnos[6:9])]
        # ---------------------------------
        
        A30_U_trials = events[events['trial_no'].isin(A30_U_indx)]
        A90_U_trials = events[events['trial_no'].isin(A90_U_indx)]
        B30_U_trials = events[events['trial_no'].isin(B30_U_indx)]
        B90_U_trials = events[events['trial_no'].isin(B90_U_indx)]
        
        # Add to onsets and durations:
        
        onsets.append(list(A30_E_trials_1[A30_E_trials_1['event_no']==7].onset))
        onsets.append(list(A30_E_trials_2[A30_E_trials_2['event_no']==7].onset))
        onsets.append(list(A30_E_trials_3[A30_E_trials_3['event_no']==7].onset))
        
        onsets.append(list(A30_U_trials[A30_U_trials['event_no']==7].onset))
        
        onsets.append(list(A90_E_trials_1[A90_E_trials_1['event_no']==7].onset))
        onsets.append(list(A90_E_trials_2[A90_E_trials_2['event_no']==7].onset))
        onsets.append(list(A90_E_trials_3[A90_E_trials_3['event_no']==7].onset))
        
        onsets.append(list(A90_U_trials[A90_U_trials['event_no']==7].onset))
        
        onsets.append(list(B30_E_trials_1[B30_E_trials_1['event_no']==7].onset))
        onsets.append(list(B30_E_trials_2[B30_E_trials_2['event_no']==7].onset))
        onsets.append(list(B30_E_trials_3[B30_E_trials_3['event_no']==7].onset))
        
        onsets.append(list(B30_U_trials[B30_U_trials['event_no']==7].onset))
        
        onsets.append(list(B90_E_trials_1[B90_E_trials_1['event_no']==7].onset))
        onsets.append(list(B90_E_trials_2[B90_E_trials_2['event_no']==7].onset))
        onsets.append(list(B90_E_trials_3[B90_E_trials_3['event_no']==7].onset))
        
        onsets.append(list(B90_U_trials[B90_U_trials['event_no']==7].onset))
        
        # ---------------------------------------------------------------
        durations.append(list(A30_E_trials_1[A30_E_trials_1['event_no']==10].onset.values - 
                              A30_E_trials_1[A30_E_trials_1['event_no']==6].onset.values))
        durations.append(list(A30_E_trials_2[A30_E_trials_2['event_no']==10].onset.values - 
                              A30_E_trials_2[A30_E_trials_2['event_no']==6].onset.values))
        durations.append(list(A30_E_trials_3[A30_E_trials_3['event_no']==10].onset.values - 
                              A30_E_trials_3[A30_E_trials_3['event_no']==6].onset.values))
        
        durations.append(list(A30_U_trials[A30_U_trials['event_no']==10].onset.values - 
                              A30_U_trials[A30_U_trials['event_no']==6].onset.values))
        
        # ---------------------------------------------------------------
        durations.append(list(A90_E_trials_1[A90_E_trials_1['event_no']==10].onset.values - 
                              A90_E_trials_1[A90_E_trials_1['event_no']==6].onset.values))
        durations.append(list(A90_E_trials_2[A90_E_trials_2['event_no']==10].onset.values - 
                              A90_E_trials_2[A90_E_trials_2['event_no']==6].onset.values))
        durations.append(list(A90_E_trials_3[A90_E_trials_3['event_no']==10].onset.values - 
                              A90_E_trials_3[A90_E_trials_3['event_no']==6].onset.values))
        
        durations.append(list(A90_U_trials[A90_U_trials['event_no']==10].onset.values - 
                              A90_U_trials[A90_U_trials['event_no']==6].onset.values))
        
        # ---------------------------------------------------------------
        durations.append(list(B30_E_trials_1[B30_E_trials_1['event_no']==10].onset.values - 
                              B30_E_trials_1[B30_E_trials_1['event_no']==6].onset.values))
        durations.append(list(B30_E_trials_2[B30_E_trials_2['event_no']==10].onset.values - 
                              B30_E_trials_2[B30_E_trials_2['event_no']==6].onset.values))
        durations.append(list(B30_E_trials_3[B30_E_trials_3['event_no']==10].onset.values - 
                              B30_E_trials_3[B30_E_trials_3['event_no']==6].onset.values))
        
        durations.append(list(B30_U_trials[B30_U_trials['event_no']==10].onset.values - 
                              B30_U_trials[B30_U_trials['event_no']==6].onset.values))
        
        # ---------------------------------------------------------------
        durations.append(list(B90_E_trials_1[B90_E_trials_1['event_no']==10].onset.values - 
                              B90_E_trials_1[B90_E_trials_1['event_no']==6].onset.values))
        durations.append(list(B90_E_trials_2[B90_E_trials_2['event_no']==10].onset.values - 
                              B90_E_trials_2[B90_E_trials_2['event_no']==6].onset.values))
        durations.append(list(B90_E_trials_3[B90_E_trials_3['event_no']==10].onset.values - 
                              B90_E_trials_3[B90_E_trials_3['event_no']==6].onset.values))
        
        durations.append(list(B90_U_trials[B90_U_trials['event_no']==10].onset.values - 
                              B90_U_trials[B90_U_trials['event_no']==6].onset.values))
        
        conditions = [c for i, c in enumerate(conditions) if onsets[i]!=[]]
        onsets = [o for o in onsets if o != []]
        durations = [d for d in durations if d != []]
        
    elif model==22:
        # events: 2 (initial scene view), 7 (first probe)
        conditions = ['A_00', #'A_00_1', 'A_00_2', 'A_00_3', 'A_00_4',
                      'A_30_exp_1', 'A_30_exp_2', 'A_30_exp_3', 'A_30_unexp',
                      'A_90_exp_1', 'A_90_exp_2', 'A_90_exp_3', 'A_90_unexp',
                      'B_00', #'B_00_1', 'B_00_2', 'B_00_3', 'B_00_4',
                      'B_30_exp_1', 'B_30_exp_2', 'B_30_exp_3', 'B_30_unexp',
                      'B_90_exp_1', 'B_90_exp_2', 'B_90_exp_3', 'B_90_unexp']
        A30mask = ((behav['InitView']==1) & (behav['FinalView']==30))
        A90mask = ((behav['InitView']==1) & (behav['FinalView']==90))
        B30mask = ((behav['InitView']==2) & (behav['FinalView']==30))
        B90mask = ((behav['InitView']==2) & (behav['FinalView']==90))
        
        A00_indx = behav.index[behav['InitView']==1]
        B00_indx = behav.index[behav['InitView']==2]
        
        A30_E_indx = behav.index[(behav['Consistent']==1) & A30mask]
        A90_E_indx = behav.index[(behav['Consistent']==1) & A90mask]
        B30_E_indx = behav.index[(behav['Consistent']==1) & B30mask]
        B90_E_indx = behav.index[(behav['Consistent']==1) & B90mask]
        
        # in this case the unexpected DON'T NEED TO BE SWAPPED:
        # Actually, they do for the A vs. B decoding - for MVPD I'm not sure
        A30_U_indx = behav.index[(behav['Consistent']==0) & B30mask]
        A90_U_indx = behav.index[(behav['Consistent']==0) & B90mask]
        B30_U_indx = behav.index[(behav['Consistent']==0) & A30mask]
        B90_U_indx = behav.index[(behav['Consistent']==0) & A90mask]
        
        # ---------------------------------
        A00_trials = events[events['trial_no'].isin(A00_indx)]
        #A00_trialnos = list(A00_E_trials.trial_no.unique())
        #A00_trialnos = random.sample(A00_E_trialnos, len(A00_trialnos))
        #A00_trials_1 = events[events['trial_no'].isin(A00_trialnos[:6])]
        #A00_trials_2 = events[events['trial_no'].isin(A00_trialnos[6:12])]
        #A00_trials_3 = events[events['trial_no'].isin(A00_trialnos[12:18])]
        #A00_trials_4 = events[events['trial_no'].isin(A00_trialnos[18:24])]
        # ---------------------------------
        B00_trials = events[events['trial_no'].isin(B00_indx)]
        #B00_trialnos = list(B00_trials.trial_no.unique())
        #B00_trialnos = random.sample(B00_trialnos, len(B00_trialnos))
        #B00_trials_1 = events[events['trial_no'].isin(B00_trialnos[:6])]
        #B00_trials_2 = events[events['trial_no'].isin(B00_trialnos[6:12])]
        #B00_trials_3 = events[events['trial_no'].isin(B00_trialnos[12:18])]
        #B00_trials_4 = events[events['trial_no'].isin(B00_trialnos[18:24])]
        # ---------------------------------
        # ---------------------------------
        A30_E_trials = events[events['trial_no'].isin(A30_E_indx)]
        A30_E_trialnos = list(A30_E_trials.trial_no.unique())
        A30_E_trialnos = random.sample(A30_E_trialnos, len(A30_E_trialnos))
        A30_E_trials_1 = events[events['trial_no'].isin(A30_E_trialnos[:3])]
        A30_E_trials_2 = events[events['trial_no'].isin(A30_E_trialnos[3:6])]
        A30_E_trials_3 = events[events['trial_no'].isin(A30_E_trialnos[6:9])]
        # ---------------------------------
        A90_E_trials = events[events['trial_no'].isin(A90_E_indx)]
        A90_E_trialnos = list(A90_E_trials.trial_no.unique())
        A90_E_trialnos = random.sample(A90_E_trialnos, len(A90_E_trialnos))
        A90_E_trials_1 = events[events['trial_no'].isin(A90_E_trialnos[:3])]
        A90_E_trials_2 = events[events['trial_no'].isin(A90_E_trialnos[3:6])]
        A90_E_trials_3 = events[events['trial_no'].isin(A90_E_trialnos[6:9])]
        # ---------------------------------
        B30_E_trials = events[events['trial_no'].isin(B30_E_indx)]
        B30_E_trialnos = list(B30_E_trials.trial_no.unique())
        B30_E_trialnos = random.sample(B30_E_trialnos, len(B30_E_trialnos))
        B30_E_trials_1 = events[events['trial_no'].isin(B30_E_trialnos[:3])]
        B30_E_trials_2 = events[events['trial_no'].isin(B30_E_trialnos[3:6])]
        B30_E_trials_3 = events[events['trial_no'].isin(B30_E_trialnos[6:9])]
        # ---------------------------------
        B90_E_trials = events[events['trial_no'].isin(B90_E_indx)]
        B90_E_trialnos = list(B90_E_trials.trial_no.unique())
        B90_E_trialnos = random.sample(B90_E_trialnos, len(B90_E_trialnos))
        B90_E_trials_1 = events[events['trial_no'].isin(B90_E_trialnos[:3])]
        B90_E_trials_2 = events[events['trial_no'].isin(B90_E_trialnos[3:6])]
        B90_E_trials_3 = events[events['trial_no'].isin(B90_E_trialnos[6:9])]
        # ---------------------------------
        
        A30_U_trials = events[events['trial_no'].isin(A30_U_indx)]
        A90_U_trials = events[events['trial_no'].isin(A90_U_indx)]
        B30_U_trials = events[events['trial_no'].isin(B30_U_indx)]
        B90_U_trials = events[events['trial_no'].isin(B90_U_indx)]
        
        # Add to onsets and durations:
        # -----------------------------------------------
        #conditions = ['A_00', #'A_00_1', 'A_00_2', 'A_00_3', 'A_00_4',
        #              'A_30_exp_1', 'A_30_exp_2', 'A_30_exp_3', 'A_30_unexp',
        #              'A_90_exp_1', 'A_90_exp_2', 'A_90_exp_3', 'A_90_unexp',
        #              'B_00', #'B_00_1', 'B_00_2', 'B_00_3', 'B_00_4',
        #              'B_30_exp_1', 'B_30_exp_2', 'B_30_exp_3', 'B_30_unexp',
        #              'B_90_exp_1', 'B_90_exp_2', 'B_90_exp_3', 'B_90_unexp']
        
        onsets.append(list(A00_trials[A00_trials['event_no']==2].onset))
        #onsets.append(list(A00_trials_1[A00_trials_1['event_no']==2].onset))
        #onsets.append(list(A00_trials_2[A00_trials_2['event_no']==2].onset))
        #onsets.append(list(A00_trials_3[A00_trials_3['event_no']==2].onset))
        #onsets.append(list(A00_trials_4[A00_trials_4['event_no']==2].onset))
        
        onsets.append(list(A30_E_trials_1[A30_E_trials_1['event_no']==7].onset))
        onsets.append(list(A30_E_trials_2[A30_E_trials_2['event_no']==7].onset))
        onsets.append(list(A30_E_trials_3[A30_E_trials_3['event_no']==7].onset))
        
        onsets.append(list(A30_U_trials[A30_U_trials['event_no']==7].onset))
        
        onsets.append(list(A90_E_trials_1[A90_E_trials_1['event_no']==7].onset))
        onsets.append(list(A90_E_trials_2[A90_E_trials_2['event_no']==7].onset))
        onsets.append(list(A90_E_trials_3[A90_E_trials_3['event_no']==7].onset))
        
        onsets.append(list(A90_U_trials[A90_U_trials['event_no']==7].onset))
        
        # ---------------------------------------------------------------------
        
        onsets.append(list(B00_trials[B00_trials['event_no']==2].onset))
        #onsets.append(list(B00_trials_1[B00_trials_1['event_no']==2].onset))
        #onsets.append(list(B00_trials_2[B00_trials_2['event_no']==2].onset))
        #onsets.append(list(B00_trials_3[B00_trials_3['event_no']==2].onset))
        #onsets.append(list(B00_trials_4[B00_trials_4['event_no']==2].onset))
        
        onsets.append(list(B30_E_trials_1[B30_E_trials_1['event_no']==7].onset))
        onsets.append(list(B30_E_trials_2[B30_E_trials_2['event_no']==7].onset))
        onsets.append(list(B30_E_trials_3[B30_E_trials_3['event_no']==7].onset))
        
        onsets.append(list(B30_U_trials[B30_U_trials['event_no']==7].onset))
        
        onsets.append(list(B90_E_trials_1[B90_E_trials_1['event_no']==7].onset))
        onsets.append(list(B90_E_trials_2[B90_E_trials_2['event_no']==7].onset))
        onsets.append(list(B90_E_trials_3[B90_E_trials_3['event_no']==7].onset))
        
        onsets.append(list(B90_U_trials[B90_U_trials['event_no']==7].onset))
        
        # Remove empty conditions:
        conditions = [c for i, c in enumerate(conditions) if onsets[i]!=[]]
        onsets = [o for o in onsets if o != []]
        
        for i, o in enumerate(onsets):
            if '00' in conditions[i]:
                durations.append([2.0] * len(o))
            else:    
                durations.append([0] * len(o))
    
    elif model==23:
        
        # events: 7 (first probe)
        conditions = ['A_30_exp_1', 'A_30_exp_2', 'A_30_exp_3', 'A_30_unexp',
                      'A_90_exp_1', 'A_90_exp_2', 'A_90_exp_3', 'A_90_unexp', 
                      'B_30_exp_1', 'B_30_exp_2', 'B_30_exp_3', 'B_30_unexp',
                      'B_90_exp_1', 'B_90_exp_2', 'B_90_exp_3', 'B_90_unexp']
        A30mask = ((behav['InitView']==1) & (behav['FinalView']==30))
        A90mask = ((behav['InitView']==1) & (behav['FinalView']==90))
        B30mask = ((behav['InitView']==2) & (behav['FinalView']==30))
        B90mask = ((behav['InitView']==2) & (behav['FinalView']==90))
        
        A30_E_indx = behav.index[(behav['Consistent']==1) & A30mask]
        A90_E_indx = behav.index[(behav['Consistent']==1) & A90mask]
        B30_E_indx = behav.index[(behav['Consistent']==1) & B30mask]
        B90_E_indx = behav.index[(behav['Consistent']==1) & B90mask]
        
        # the unexpected ones NEED TO BE SWAPPED:
        # (unexpected stimulus came from the other initial viewpoint)
        A30_U_indx = behav.index[(behav['Consistent']==0) & B30mask]
        A90_U_indx = behav.index[(behav['Consistent']==0) & B90mask]
        B30_U_indx = behav.index[(behav['Consistent']==0) & A30mask]
        B90_U_indx = behav.index[(behav['Consistent']==0) & A90mask]
        
        # ---------------------------------
        A30_E_trials = events[events['trial_no'].isin(A30_E_indx)]
        A30_E_trialnos = list(A30_E_trials.trial_no.unique())
        random.seed(0)
        A30_E_trialnos = random.sample(A30_E_trialnos, len(A30_E_trialnos))
        A30_E_trials_1 = events[events['trial_no'].isin(A30_E_trialnos[:3])]
        A30_E_trials_2 = events[events['trial_no'].isin(A30_E_trialnos[3:6])]
        A30_E_trials_3 = events[events['trial_no'].isin(A30_E_trialnos[6:9])]
        # ---------------------------------
        A90_E_trials = events[events['trial_no'].isin(A90_E_indx)]
        A90_E_trialnos = list(A90_E_trials.trial_no.unique())
        random.seed(0)
        A90_E_trialnos = random.sample(A90_E_trialnos, len(A90_E_trialnos))
        A90_E_trials_1 = events[events['trial_no'].isin(A90_E_trialnos[:3])]
        A90_E_trials_2 = events[events['trial_no'].isin(A90_E_trialnos[3:6])]
        A90_E_trials_3 = events[events['trial_no'].isin(A90_E_trialnos[6:9])]
        # ---------------------------------
        B30_E_trials = events[events['trial_no'].isin(B30_E_indx)]
        B30_E_trialnos = list(B30_E_trials.trial_no.unique())
        random.seed(0)
        B30_E_trialnos = random.sample(B30_E_trialnos, len(B30_E_trialnos))
        B30_E_trials_1 = events[events['trial_no'].isin(B30_E_trialnos[:3])]
        B30_E_trials_2 = events[events['trial_no'].isin(B30_E_trialnos[3:6])]
        B30_E_trials_3 = events[events['trial_no'].isin(B30_E_trialnos[6:9])]
        # ---------------------------------
        B90_E_trials = events[events['trial_no'].isin(B90_E_indx)]
        B90_E_trialnos = list(B90_E_trials.trial_no.unique())
        random.seed(0)
        B90_E_trialnos = random.sample(B90_E_trialnos, len(B90_E_trialnos))
        B90_E_trials_1 = events[events['trial_no'].isin(B90_E_trialnos[:3])]
        B90_E_trials_2 = events[events['trial_no'].isin(B90_E_trialnos[3:6])]
        B90_E_trials_3 = events[events['trial_no'].isin(B90_E_trialnos[6:9])]
        # ---------------------------------
        
        A30_U_trials = events[events['trial_no'].isin(A30_U_indx)]
        A90_U_trials = events[events['trial_no'].isin(A90_U_indx)]
        B30_U_trials = events[events['trial_no'].isin(B30_U_indx)]
        B90_U_trials = events[events['trial_no'].isin(B90_U_indx)]
        
        # Add to onsets and durations:
        
        onsets.append(list(A30_E_trials_1[A30_E_trials_1['event_no']==7].onset))
        onsets.append(list(A30_E_trials_2[A30_E_trials_2['event_no']==7].onset))
        onsets.append(list(A30_E_trials_3[A30_E_trials_3['event_no']==7].onset))
        
        onsets.append(list(A30_U_trials[A30_U_trials['event_no']==7].onset))
        
        onsets.append(list(A90_E_trials_1[A90_E_trials_1['event_no']==7].onset))
        onsets.append(list(A90_E_trials_2[A90_E_trials_2['event_no']==7].onset))
        onsets.append(list(A90_E_trials_3[A90_E_trials_3['event_no']==7].onset))
        
        onsets.append(list(A90_U_trials[A90_U_trials['event_no']==7].onset))
        
        onsets.append(list(B30_E_trials_1[B30_E_trials_1['event_no']==7].onset))
        onsets.append(list(B30_E_trials_2[B30_E_trials_2['event_no']==7].onset))
        onsets.append(list(B30_E_trials_3[B30_E_trials_3['event_no']==7].onset))
        
        onsets.append(list(B30_U_trials[B30_U_trials['event_no']==7].onset))
        
        onsets.append(list(B90_E_trials_1[B90_E_trials_1['event_no']==7].onset))
        onsets.append(list(B90_E_trials_2[B90_E_trials_2['event_no']==7].onset))
        onsets.append(list(B90_E_trials_3[B90_E_trials_3['event_no']==7].onset))
        
        onsets.append(list(B90_U_trials[B90_U_trials['event_no']==7].onset))
        
        conditions = [c for i, c in enumerate(conditions) if onsets[i]!=[]]
        onsets = [o for o in onsets if o != []]
        
        for o in onsets:
            durations.append([0] * len(o))
    
    elif model==27:
        # events: 7 (first probe)
        conditions = ['expected_1', 'expected_2', 'expected_3', 'unexpected']
        expindx = behav.index[behav['Consistent']==1]
        exptrials = events[events['trial_no'].isin(expindx)]
        unexptrials = events[~events['trial_no'].isin(expindx)]
        
        exp_trialnos = list(exptrials.trial_no.unique())
        random.seed(0)
        exp_trialnos = random.sample(exp_trialnos, len(exp_trialnos))
        exp_trials_1 = events[events['trial_no'].isin(exp_trialnos[:12])]
        exp_trials_2 = events[events['trial_no'].isin(exp_trialnos[12:24])]
        exp_trials_3 = events[events['trial_no'].isin(exp_trialnos[24:36])]
        
        onsets.append(list(exp_trials_1[exp_trials_1['event_no']==7].onset))
        onsets.append(list(exp_trials_2[exp_trials_2['event_no']==7].onset))
        onsets.append(list(exp_trials_3[exp_trials_3['event_no']==7].onset))
        onsets.append(list(unexptrials[unexptrials['event_no']==7].onset))
        for o in onsets:
            durations.append([0] * len(o))
            
    elif model in [28, 31, 33, 35]:
        
        subjid = re.search(r"sub-\d+", eventsfile).group(0)
        runno = re.search(r"run-\d+", eventsfile).group(0)
        
        if model==28:
            random.seed(subjid+runno)
        elif model==31:
            random.seed((subjid+runno)*2)
        elif model==33:
            random.seed((subjid+runno)*3)
        elif model==35:
            random.seed((subjid+runno)*4)
        
        # events: 7 (first probe)
        conditions = ['A_30_exp_1', 'A_30_exp_2', 'A_30_exp_3', 'A_30_unexp',
                      'A_90_exp_1', 'A_90_exp_2', 'A_90_exp_3', 'A_90_unexp', 
                      'B_30_exp_1', 'B_30_exp_2', 'B_30_exp_3', 'B_30_unexp',
                      'B_90_exp_1', 'B_90_exp_2', 'B_90_exp_3', 'B_90_unexp']
        A30mask = ((behav['InitView']==1) & (behav['FinalView']==30))
        A90mask = ((behav['InitView']==1) & (behav['FinalView']==90))
        B30mask = ((behav['InitView']==2) & (behav['FinalView']==30))
        B90mask = ((behav['InitView']==2) & (behav['FinalView']==90))
        
        A30_E_indx = behav.index[(behav['Consistent']==1) & A30mask]
        A90_E_indx = behav.index[(behav['Consistent']==1) & A90mask]
        B30_E_indx = behav.index[(behav['Consistent']==1) & B30mask]
        B90_E_indx = behav.index[(behav['Consistent']==1) & B90mask]
        
        # the unexpected ones NEED TO BE SWAPPED:
        # (unexpected stimulus came from the other initial viewpoint)
        A30_U_indx = behav.index[(behav['Consistent']==0) & B30mask]
        A90_U_indx = behav.index[(behav['Consistent']==0) & B90mask]
        B30_U_indx = behav.index[(behav['Consistent']==0) & A30mask]
        B90_U_indx = behav.index[(behav['Consistent']==0) & A90mask]
        
        # ---------------------------------
        A30_E_trials = events[events['trial_no'].isin(A30_E_indx)]
        A30_E_trialnos = list(A30_E_trials.trial_no.unique())
        A30_E_trialnos = random.sample(A30_E_trialnos, len(A30_E_trialnos))
        A30_E_trials_1 = events[events['trial_no'].isin(A30_E_trialnos[:3])]
        A30_E_trials_2 = events[events['trial_no'].isin(A30_E_trialnos[3:6])]
        A30_E_trials_3 = events[events['trial_no'].isin(A30_E_trialnos[6:9])]
        # ---------------------------------
        A90_E_trials = events[events['trial_no'].isin(A90_E_indx)]
        A90_E_trialnos = list(A90_E_trials.trial_no.unique())
        A90_E_trialnos = random.sample(A90_E_trialnos, len(A90_E_trialnos))
        A90_E_trials_1 = events[events['trial_no'].isin(A90_E_trialnos[:3])]
        A90_E_trials_2 = events[events['trial_no'].isin(A90_E_trialnos[3:6])]
        A90_E_trials_3 = events[events['trial_no'].isin(A90_E_trialnos[6:9])]
        # ---------------------------------
        B30_E_trials = events[events['trial_no'].isin(B30_E_indx)]
        B30_E_trialnos = list(B30_E_trials.trial_no.unique())
        B30_E_trialnos = random.sample(B30_E_trialnos, len(B30_E_trialnos))
        B30_E_trials_1 = events[events['trial_no'].isin(B30_E_trialnos[:3])]
        B30_E_trials_2 = events[events['trial_no'].isin(B30_E_trialnos[3:6])]
        B30_E_trials_3 = events[events['trial_no'].isin(B30_E_trialnos[6:9])]
        # ---------------------------------
        B90_E_trials = events[events['trial_no'].isin(B90_E_indx)]
        B90_E_trialnos = list(B90_E_trials.trial_no.unique())
        B90_E_trialnos = random.sample(B90_E_trialnos, len(B90_E_trialnos))
        B90_E_trials_1 = events[events['trial_no'].isin(B90_E_trialnos[:3])]
        B90_E_trials_2 = events[events['trial_no'].isin(B90_E_trialnos[3:6])]
        B90_E_trials_3 = events[events['trial_no'].isin(B90_E_trialnos[6:9])]
        # ---------------------------------
        
        A30_U_trials = events[events['trial_no'].isin(A30_U_indx)]
        A90_U_trials = events[events['trial_no'].isin(A90_U_indx)]
        B30_U_trials = events[events['trial_no'].isin(B30_U_indx)]
        B90_U_trials = events[events['trial_no'].isin(B90_U_indx)]
        
        # Add to onsets and durations:
        
        onsets.append(list(A30_E_trials_1[A30_E_trials_1['event_no']==7].onset))
        onsets.append(list(A30_E_trials_2[A30_E_trials_2['event_no']==7].onset))
        onsets.append(list(A30_E_trials_3[A30_E_trials_3['event_no']==7].onset))
        
        onsets.append(list(A30_U_trials[A30_U_trials['event_no']==7].onset))
        
        onsets.append(list(A90_E_trials_1[A90_E_trials_1['event_no']==7].onset))
        onsets.append(list(A90_E_trials_2[A90_E_trials_2['event_no']==7].onset))
        onsets.append(list(A90_E_trials_3[A90_E_trials_3['event_no']==7].onset))
        
        onsets.append(list(A90_U_trials[A90_U_trials['event_no']==7].onset))
        
        onsets.append(list(B30_E_trials_1[B30_E_trials_1['event_no']==7].onset))
        onsets.append(list(B30_E_trials_2[B30_E_trials_2['event_no']==7].onset))
        onsets.append(list(B30_E_trials_3[B30_E_trials_3['event_no']==7].onset))
        
        onsets.append(list(B30_U_trials[B30_U_trials['event_no']==7].onset))
        
        onsets.append(list(B90_E_trials_1[B90_E_trials_1['event_no']==7].onset))
        onsets.append(list(B90_E_trials_2[B90_E_trials_2['event_no']==7].onset))
        onsets.append(list(B90_E_trials_3[B90_E_trials_3['event_no']==7].onset))
        
        onsets.append(list(B90_U_trials[B90_U_trials['event_no']==7].onset))
        
        conditions = [c for i, c in enumerate(conditions) if onsets[i]!=[]]
        onsets = [o for o in onsets if o != []]
        
        for o in onsets:
            durations.append([0] * len(o))
            
    elif model in [30, 32, 34, 36]:
        
        subjid = re.search(r"sub-\d+", eventsfile).group(0)
        runno = re.search(r"run-\d+", eventsfile).group(0)
        if model == 30:
            random.seed(subjid+runno)
        elif model == 32:
            random.seed((subjid+runno)*2)
        elif model == 34:
            random.seed((subjid+runno)*3)
        elif model == 36:
            random.seed((subjid+runno)*4)
        
        # events: 7 (first probe)
        conditions = ['expected_1', 'expected_2', 'expected_3', 'unexpected']
        expindx = behav.index[behav['Consistent']==1]
        exptrials = events[events['trial_no'].isin(expindx)]
        unexptrials = events[~events['trial_no'].isin(expindx)]
        
        exp_trialnos = list(exptrials.trial_no.unique())
        exp_trialnos = random.sample(exp_trialnos, len(exp_trialnos))
        exp_trials_1 = events[events['trial_no'].isin(exp_trialnos[:12])]
        exp_trials_2 = events[events['trial_no'].isin(exp_trialnos[12:24])]
        exp_trials_3 = events[events['trial_no'].isin(exp_trialnos[24:36])]
        
        onsets.append(list(exp_trials_1[exp_trials_1['event_no']==7].onset))
        onsets.append(list(exp_trials_2[exp_trials_2['event_no']==7].onset))
        onsets.append(list(exp_trials_3[exp_trials_3['event_no']==7].onset))
        onsets.append(list(unexptrials[unexptrials['event_no']==7].onset))
        for o in onsets:
            durations.append([0] * len(o))
            
    elif model==37:
        
        subjid = re.search(r"sub-\d+", eventsfile).group(0)
        runno = re.search(r"run-\d+", eventsfile).group(0)
        
        random.seed(subjid+runno)
        
        # events: 7 (first probe)
        conditions = ['A_90_exp', 'A_90_unexp', 
                      'B_90_exp', 'B_90_unexp']
        
        A90mask = ((behav['InitView']==1) & (behav['FinalView']==90) & (behav['Sequence_2']==15))
        B90mask = ((behav['InitView']==2) & (behav['FinalView']==90) & (behav['Sequence_2']==15))
        
        A90_E_indx = behav.index[(behav['Consistent']==1) & A90mask]
        B90_E_indx = behav.index[(behav['Consistent']==1) & B90mask]
        
        # the unexpected ones NEED TO BE SWAPPED:
        # (unexpected stimulus came from the other initial viewpoint)
        A90_U_indx = behav.index[(behav['Consistent']==0) & B90mask]
        B90_U_indx = behav.index[(behav['Consistent']==0) & A90mask]
        n_trials = min(len(A90_U_indx), len(B90_U_indx))
        
        # ---------------------------------
        A90_E_trialnos = random.sample(list(A90_E_indx), n_trials)
        A90_E_trials = events[events['trial_no'].isin(A90_E_trialnos)]
        # ---------------------------------
        B90_E_trialnos = random.sample(list(B90_E_indx), n_trials)
        B90_E_trials = events[events['trial_no'].isin(B90_E_trialnos)]
        # ---------------------------------
        A90_U_trialnos = random.sample(list(A90_U_indx), n_trials)
        A90_U_trials = events[events['trial_no'].isin(A90_U_trialnos)]
        # ---------------------------------
        B90_U_trialnos = random.sample(list(B90_U_indx), n_trials)
        B90_U_trials = events[events['trial_no'].isin(B90_U_trialnos)]
        
        # Add to onsets and durations:
        
        onsets.append(list(A90_E_trials[A90_E_trials['event_no']==7].onset))
        onsets.append(list(A90_U_trials[A90_U_trials['event_no']==7].onset))
        
        onsets.append(list(B90_E_trials[B90_E_trials['event_no']==7].onset))
        onsets.append(list(B90_U_trials[B90_U_trials['event_no']==7].onset))
        
        conditions = [c for i, c in enumerate(conditions) if onsets[i]!=[]]
        onsets = [o for o in onsets if o != []]
        
        for o in onsets:
            durations.append([0] * len(o))
            
    else:
        raise ValueError('Model {:g} unknown!'.format(model))
        
        
    evs = Bunch(conditions=conditions, onsets=onsets, durations=durations)
        
    return evs