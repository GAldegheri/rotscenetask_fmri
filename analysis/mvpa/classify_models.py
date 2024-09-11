# =================================================================================================
# Helper functions to determine which models are split in wide/narrow, exp/unexp, thirds
# =================================================================================================
    
def isWideNarrow(opt):
    # Little helper function to determine
    # which tasks/models should be labeled as 
    # wide vs. narrow
    
    if (opt.task=='train' and opt.model in [2, 3]) or (opt.task=='test' and opt.model in [3, 4, 6, 7, 8, 10, 11]):
        return True
    else:
        return False
    
# -------------------------------------------------------------------------------------------------

def isExpUnexp(opt):
    # Little helper function to determine
    # which tasks/models should be split into 
    # expected and unexpected
    
    if opt.task=='test' and opt.model in [5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 19, 
                                          20, 21, 22, 23, 24, 28, 29, 30, 31]:
        return True
    else:
        return False

# -------------------------------------------------------------------------------------------------

def DivideInThirds(opt):
    # Should this task/model be
    # divided in thirds?
    
    if opt.task=='test' and opt.model in [7, 9, 11, 12, 13, 15, 19, 20, 21, 22, 
                                          23, 24, 28, 29, 30, 31]:
        return True
    else:
        return False

# -------------------------------------------------------------------------------------------------

def is30or90(opt):
    # Little helper function to determine
    # which tasks/models should be labeled as 
    # rot. 30 vs. 90 degrees
    
    if (opt.task=='train' and opt.model in [1, 6]) or (opt.task=='test' and opt.model in [2, 13, 14]):
        return True
    else:
        return False
# -------------------------------------------------------------------------------------------------
 
def isAorB(opt):
    if (opt.task=='train' and opt.model==7) or (opt.task=='test' and opt.model in [19, 22]):
        return True
    else:
        return False

# -------------------------------------------------------------------------------------------------

def isAllViews(opt):
    # Which tasks/models contain both object view (A/B)
    # and scene rotation (30/90).
    
    if (opt.task=='train' and opt.model in [4, 5]) or (opt.task=='test' and opt.model in [12, 15, 16, 17, 20, 21, 23, 24,
                                                                                          28, 29, 31]):
        return True
    else:
        return False
    
# -------------------------------------------------------------------------------------------------

def isInitAndFinal(opt):
    # Models (for now only one) which include both initial view and final
    return (opt.task=='test' and opt.model==22)