from mvpa.loading import load_betas
import os
from general_utils import Options
from configs import bids_dir


if __name__ == "__main__":
    opt = Options(sub='sub-001',task='test',model=15)
    mask_templ = os.path.join(bids_dir, 'derivatives', 'spm-preproc', 'derivatives', 
                            'roi-masks', '{:s}', '{:s}_ba-17-18_R_contr-objscrvsbas_top-1000.nii')

    firDS = load_betas(opt, fir=True, mask_templ=mask_templ, max_delay=5)