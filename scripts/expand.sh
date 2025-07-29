#!/bin/bash
#SBATCH -A p30041
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 50
#SBATCH --mem=0
#SBATCH -t 4:00:00
#SBATCH --job-name="expand"
#SBATCH --output=/home/spn1560/coarse-grain-rxns/logs/out/%A
#SBATCH --error=/home/spn1560/coarse-grain-rxns/logs/error/%A
#SBATCH --array=0-10
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu

# Args
script=/home/spn1560/coarse-grain-rxns/scripts/expand.py
rules_sweep=(
    "mechinformed_rules_w_coreactants"
    "mechinferred_dt_01_rules_w_coreactants" 
    "mechinferred_dt_02_rules_w_coreactants"
    "mechinferred_dt_04_rules_w_coreactants"
    "mechinferred_dt_13_rules_w_coreactants"
    "mechinferred_dt_91_rules_w_coreactants"
    "rc_plus_0_rules_w_coreactants" 
    "rc_plus_1_rules_w_coreactants"
    "rc_plus_2_rules_w_coreactants"
    "rc_plus_3_rules_w_coreactants"
    "rc_plus_4_rules_w_coreactants"
)
processes=50 # MAKE SURE THIS MATCHES -n above

# Commands
ulimit -c 0
module purge
source /home/spn1560/coarse-grain-rxns/.venv/bin/activate
python $script processes=$processes rules=${rules_sweep[$SLURM_ARRAY_TASK_ID]}