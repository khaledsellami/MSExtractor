#!/bin/bash
#SBATCH --time=24:0:00
#SBATCH --nodes=1 # number of nodes
#SBATCH --mem=16000 # memory per node
#SBATCH --cpus-per-task=40
#SBATCH --array=0-4
#SBATCH --output=./logs/$APP/slurm_log_%A_%a.out
source ${ENV_PATH}
python ./cc_cli.py $APP --data ${DATA_PATH} -v -j $SLURM_ARRAY_TASK_ID -n $SLURM_ARRAY_JOB_ID -M ${MAX_MNM} -m ${MIN_MNM}
