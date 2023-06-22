import numpy as np

################################# Add motion parameters #####################################

def read_motion_par(mp_files, task, run):
    '''
    mp_files is a list of .txt motion parameter files
    (more than 1 only for subjects who got out of the scanner)
    '''
    
    start_dict = {
        'test': ([12, 417, 1475, 1880, 2619, 3024, 4082], 404),
        'train': ([822, 2285, 3429], 333),
        'funcloc': ([1156, 3763], 318)
    }
    
    if isinstance(mp_files, str):
        mp_files = [mp_files]
    
    # outputs list of lists (one per regressor a.k.a. column)
    motionarray = np.empty((0, 6))
    for f in mp_files:
        motionarray = np.append(motionarray, np.loadtxt(f), axis=0)
    
    thisstart = start_dict[task][0][run] # scalar: start of run
    thislength = start_dict[task][1]
    if motionarray.shape==(4476, 6): # subj. 14 missed the 10 inverted bold scans
        thisstart -= 10
        
    allmotionregs = list(motionarray[thisstart:thisstart+thislength])
    allmotionregs = np.array(allmotionregs)
    
    return [list(col) for col in allmotionregs.T] # this *should* be what Nipype wants

# ---------------------------------------------------------------------------------

def Add_Motion_Regressors(subj_info, task, use_motion_reg, motpar):
    '''
    - subj_info: output of ModelSpecify
    - use_motion_reg: bool
    - motpar: list of .txt files
    '''
    import sys
    sys.path.append('/project/3018040.05/rotscenetask_fmri/analysis/')
    from mvpa.motionparameters import read_motion_par
    
    if use_motion_reg:
        for run, _ in enumerate(subj_info):
            subj_info[run].regressor_names = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
            subj_info[run].regressors = read_motion_par(motpar, task, run) # list of 6 columns
    
    return subj_info