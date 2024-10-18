from torque.torque_funcs import submit_job

if __name__=="__main__":
    subjlist = [f'sub-{i:03d}' for i in range(17, 36)]
    #subjlist = ['sub-010']
    roilist = [f"ba-17-18_contr-objscrvsbas_top-{v:g}_nothresh" for v in [500, 600, 700, 800, 900, 1000]]
    for s in subjlist:
        for r in roilist:
            submit_job("/project/3018040.05/rotscenetask_fmri/analysis/infocoupling_FIR_main_nonipype.py", 
                    {"sub": s, "roi": r}, "run_infoFIR")
        # submit_job("/project/3018040.05/rotscenetask_fmri/analysis/readresults/readres_univar.py",
        #            {"sub": s}, "save_univar")