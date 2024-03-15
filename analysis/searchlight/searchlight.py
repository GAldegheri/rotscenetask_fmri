import numpy as np

# =================================================================================================
# Searchlight post-processing:
# =================================================================================================

def extract_accuracy(res):
    
    # res should be a single 2D array
    
    return np.mean(res[:, 0]==res[:, 1]) # whether outputs match targets

# -------------------------------------------------------------------------------------------------

def extract_distance(res):
    
    distances = res[:, 2].astype(float)
    zscoredists = (distances - np.mean(distances))/np.std(distances)
    # which labels correspond to +1 and -1:
    poslabel = np.unique(res[:, 0][np.sign(zscoredists)==1.])[0] # which output corresponds to a positive distance
    neglabel = np.unique(res[:, 0][np.sign(zscoredists)==-1.])[0]
    truelabelvec = np.zeros(zscoredists.shape)
    truelabelvec[res[:,1]==poslabel]=1.
    truelabelvec[res[:,1]==neglabel]=-1.
    
    return np.mean(zscoredists*truelabelvec)

# -------------------------------------------------------------------------------------------------

def postproc_searchlight(res, measure='accuracy'):
    
    if measure=='accuracy':
        extractfunc = extract_accuracy
    elif measure=='distance':
        extractfunc = extract_distance
        
    if isinstance(res, tuple): # exp./unexp.
        
        # recursive
        #return postproc_searchlight(res[0]) - postproc_searchlight(res[1])
        return np.vstack([postproc_searchlight(res[0], measure), postproc_searchlight(res[1], measure)])
    
    if res.ndim==3: # splits of expected trials, A vs. B etc.
        acc = 0
        for i in range(0, res.shape[2]):
            acc += extractfunc(res[:, :, i])
        acc /= res.shape[2]
        
    elif isinstance(res, np.ndarray):
        acc = extractfunc(res)
        
    elif isinstance(res, float):
        acc = res
    
    return acc