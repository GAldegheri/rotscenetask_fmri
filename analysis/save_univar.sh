#!/bin/bash
#PBS -N save_univar.sh
#PBS -l walltime=02:00:00,mem=16g
#PBS -l nodes=1:ppn=1
#PBS -q batch
#PBS -o torque/jobs/output/save_univar.sh_output.txt
#PBS -e torque/jobs/output/save_univar.sh_error.txt
cd /project/3018040.05/rotscenetask_fmri/analysis/
source activate giacomo37
python /project/3018040.05/rotscenetask_fmri/analysis/readresults/readres_univar.py