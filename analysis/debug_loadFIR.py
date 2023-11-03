from mvpa.loading import load_betas
import os
from utils import Options
from configs import bids_dir
from mvpa.mvpa_utils import correct_labels
import pdb

if __name__ == "__main__":
    train_opt = Options(sub='sub-001', task='train', model=5)
    test_opt = Options(sub='sub-001',task='test',model=15)
    mask_templ = os.path.join(bids_dir, 'derivatives', 'spm-preproc', 'derivatives', 
                            'roi-masks', '{:s}', '{:s}_ba-17-18_R_contr-objscrvsbas_top-500.nii')

    trainDS = load_betas(train_opt, mask_templ=mask_templ, fir=False)
    trainDS = correct_labels(trainDS, train_opt)
    testDS = load_betas(test_opt, fir=True, mask_templ=mask_templ, max_delay=15)
    testDS = correct_labels(testDS, test_opt)
    pdb.set_trace()