import pandas as pd

################################## Single-trial betas #######################################

def specify_trialbetas(trialinfo):
    '''
    Input: trialinfo - Bunch containing:
                        - nifti
                        - evfile
                        - trial
                        - event
                        - motion
    -------------------------------------------
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

    ** = occluder present

    '''
    
    import pandas as pd
    import numpy as np
    from nipype.interfaces.base import Bunch
    
    events = pd.read_csv(trialinfo.evfile, sep='\t')
    
    tr = trialinfo.trial
    ev = trialinfo.event
    
    conditions = ['thistrial', 'othertrials']
    #exclude_events = [8, 9, 10, 11, 12] # to be excluded from 'other' list because they shouldn't
                                # be decorrelated - they're basically continuations of the event of interest!
                                # maybe add 11 & 12 too?
    
    include_events = [2, 7]
    
    thistrial = events[events['trial_no']==tr]

    # two relevant events: initial view (2) and final view (7)

    theseonsets = [o for o in thistrial[thistrial['event_no']==ev].onset.values]
    thesedurations = []

    if ev==2: # initial view (event 2, lasts until onset of event 3)
        thesedurations.append(thistrial[thistrial['event_no']==3].onset.values[0] - thistrial[thistrial['event_no']==2].onset.values[0])

    elif ev==7: # final view (event 7, duration 0 - very brief)
        thesedurations.append(0)

    # loop through all other trials
    otheronsets = []
    otherdurations = []
    for othertr in events.trial_no.unique():
        for otherev in include_events: #events.event_no.unique():
            if not ((tr==othertr) & (ev==otherev)):
                # everything except this particular trial and event
                othertrial = events[events['trial_no']==othertr]
                otherons = othertrial[othertrial['event_no']==otherev].onset.values[0]

                if not np.isnan(otherons):
                    otheronsets.append(otherons)
                    if otherev != np.max(events.event_no.unique()):
                        otherdur = othertrial[othertrial['event_no']==otherev+1].onset.values[0] - othertrial[othertrial['event_no']==otherev].onset.values[0]
                    elif othertr != np.max(events.trial_no.unique()):
                        otherdur = events[(events['trial_no']==othertr+1)&(events['event_no']==1)].onset.values - othertrial[othertrial['event_no']==otherev].onset.values
                    else: # Last event of last trial
                        otherdur = 0

                    if otherdur < 1.0 or np.isnan(otherdur): 
                        otherdur = 0
                    otherdurations.append(otherdur)

    onsets = [theseonsets, otheronsets]
    durations = [thesedurations, otherdurations]
                            
    evs = Bunch(conditions=conditions, onsets=onsets, durations=durations)

    return [evs], [trialinfo.nifti], trialinfo.motion, tr, ev

# ---------------------------------------------------------------------------------

def get_first_beta(beta_images):
    '''
    Stupid utility to get the first of a list of beta images
    '''
    
    return beta_images[0]

# ---------------------------------------------------------------------------------

def rename_trialbetas(in_file, trialno, eventno):
    
    import os
    from glob import glob
    
    newfile = 'trial-{:03d}_ev-{:g}.nii'.format(trialno, eventno)
    newfile = os.path.join(os.path.split(in_file)[0], newfile)
    os.rename(in_file, newfile)
    
    # remove useless betas:
    allbetas = glob(os.path.join(os.path.split(in_file)[0], 'beta_*.nii'))
    if len(allbetas) != 0:
        for b in allbetas:
            os.remove(b)
    
    return newfile