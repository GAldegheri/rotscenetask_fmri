from torque.torque_funcs import submit_job

if __name__=="__main__":
    subjlist = [f'sub-{i:03d}' for i in range(4, 36)]
    for s in subjlist:
        submit_job("/project/3018040.05/rotscenetask_fmri/analysis/infocoupling_FIR_main_nonipype.py", 
                   {"sub": s, "roi": "ba-17-18_contr-objscrvsbas_top-500"}, "run_infoFIR")