#!/bin/bash
#SBATCH -A b1039
#SBATCH -p b1039
#SBATCH -N 1
#SBATCH -n 50
#SBATCH --mem=80GB
#SBATCH -t 48:00:00
#SBATCH --job-name="expand"
#SBATCH --output=/home/spn1560/coarse-grain-rxns/logs/out/%x_%A_%a.out
#SBATCH --error=/home/spn1560/coarse-grain-rxns/logs/error/%x_%A_%a.err
#SBATCH --array=0-2
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu

# Args
script=/home/spn1560/coarse-grain-rxns/scripts/expand.py
rules_sweep=(
    evodex_Cm_rules_w_coreactants
    evodex_Dm_rules_w_coreactants
    evodex_Em_rules_w_coreactants
    # evodex_Cm_rules_before_2015_w_coreactants
    # evodex_Dm_rules_before_2015_w_coreactants
    # evodex_Em_rules_before_2015_w_coreactants
    # mechinferred_dt_035_rules_before_2015_direct_mcsa_only_w_coreactants
    # mechinferred_dt_059_rules_before_2015_direct_mcsa_only_w_coreactants
    # mechinferred_dt_106_rules_before_2015_direct_mcsa_only_w_coreactants
    # mechinferred_dt_244_rules_before_2015_direct_mcsa_only_w_coreactants
    # mechinferred_dt_961_rules_before_2015_direct_mcsa_only_w_coreactants
    # mechinferred_dt_956_rules_w_coreacatants
    # mechinferred_dt_224_rules_w_coreacatants
    # mechinferred_dt_112_rules_w_coreacatants
    # mechinferred_dt_039_rules_w_coreacatants
    # mechinferred_dt_019_rules_w_coreacatants
    # mechinformed_rules_w_coreacatants
    # imt_rules_w_coreactants
    # rdchiral_rules_w_coreactants
    # rc_plus_0_rules_w_coreactants
    # rc_plus_1_rules_w_coreactants
    # rc_plus_2_rules_w_coreactants
    # rc_plus_3_rules_w_coreactants
    # rc_plus_4_rules_w_coreactants
    # retrobiocat_rules_w_coreactants
)
starters=250728_benchmark_starters
generations=2
explicit_h=true
processes=50 # MAKE SURE THIS MATCHES -n above

# Commands
ulimit -c 0
module purge
source /home/spn1560/coarse-grain-rxns/.venv/bin/activate
python $script starters=$starters generations=$generations processes=$processes rules=${rules_sweep[$SLURM_ARRAY_TASK_ID]} explicit_h=$explicit_h
