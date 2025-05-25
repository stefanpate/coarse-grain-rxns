#!/bin/bash
#SBATCH -A b1039
#SBATCH -p b1039
#SBATCH -N 1
#SBATCH -n 50
#SBATCH --mem=0
#SBATCH -t 3:00:00
#SBATCH --job-name="expand"
#SBATCH --output=/home/spn1560/coarse-grain-rxns/logs/out/%A
#SBATCH --error=/home/spn1560/coarse-grain-rxns/logs/error/%A
#SBATCH --array=0-9
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu

# Args
script=/home/spn1560/coarse-grain-rxns/scripts/expand.py
rules_sweep=(
    "mechinformed_rules_w_coreactants"
    "imt_rules_w_coreactants"
    "mechinferred_dt_2_rules_w_coreactants" 
    "mechinferred_dt_3_rules_w_coreactants"
    "mechinferred_dt_6_rules_w_coreactants"
    "mechinferred_dt_15_rules_w_coreactants"
    "mechinferred_dt_98_rules_w_coreactants"
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