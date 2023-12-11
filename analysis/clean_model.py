import os
import shutil

def main(task, model):
    betadir = '../../bids/derivatives/spm-preproc/derivatives/spm-stats/betas/'
    contrdir = '../../bids/derivatives/spm-preproc/derivatives/spm-stats/contrasts/'
    fir = True
    contrasts = False
    subjlist = [f'sub-{i:03d}' for i in range(1, 36)]
    for s in subjlist:
        thisbetadir = os.path.join(betadir, s, task, f'model_{model}')
        if fir:
            thisbetadir = os.path.join(thisbetadir, 'FIR')
        if os.path.isdir(thisbetadir):
            shutil.rmtree(thisbetadir)
            print(thisbetadir, 'removed.')
        else:
            print(thisbetadir, 'not found.')
        if contrasts:
            thiscontrdir = os.path.join(contrdir, s, task, f'model_{model}')
            if os.path.isdir(thiscontrdir):
                shutil.rmtree(thiscontrdir)
                print(thiscontrdir, 'removed.')
            else:
                print(thisbetadir, 'not found.')
                
if __name__=="__main__":
    main('test', 30)
        