#!/bin/bash
#SBATCH -A p30041
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 50
#SBATCH --mem=80GB
#SBATCH -t 18:00:00
#SBATCH --job-name="expand"
#SBATCH --output=/home/spn1560/coarse-grain-rxns/logs/out/%x_%A_%a.out
#SBATCH --error=/home/spn1560/coarse-grain-rxns/logs/error/%x_%A_%a.err
#SBATCH --array=0-12
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu

# Args
script=/home/spn1560/coarse-grain-rxns/scripts/expand.py
rules_sweep=(
    rc_plus_0_rules_before_2015_w_coreactants
    rc_plus_1_rules_before_2015_w_coreactants
    rc_plus_2_rules_before_2015_w_coreactants
    rc_plus_3_rules_before_2015_w_coreactants
    rc_plus_4_rules_before_2015_w_coreactants
    mechinferred_dt_005_rules_before_2015_w_coreactants
    mechinferred_dt_009_rules_before_2015_w_coreactants
    mechinferred_dt_021_rules_before_2015_w_coreactants
    mechinferred_dt_069_rules_before_2015_w_coreactants
    mechinferred_dt_932_rules_before_2015_w_coreactants
    mechinformed_rules_before_2015_w_coreactants
    imt_rules_before_2015_w_coreactants
    rdchiral_rules_before_2015_w_coreactants
)
starters=after_2015_cpds
generations=1
processes=50 # MAKE SURE THIS MATCHES -n above

# Commands
ulimit -c 0
module purge
source /home/spn1560/coarse-grain-rxns/.venv/bin/activate
python $script starters=$starters generations=$generations processes=$processes rules=${rules_sweep[$SLURM_ARRAY_TASK_ID]}