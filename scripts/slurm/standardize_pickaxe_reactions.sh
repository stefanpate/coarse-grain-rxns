#!/bin/bash
#SBATCH -A p30041
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=20G
#SBATCH -t 4:00:00
#SBATCH --job-name="std_pk_rxns"
#SBATCH --output=/home/spn1560/coarse-grain-rxns/logs/out/%x_%A_%a.out
#SBATCH --error=/home/spn1560/coarse-grain-rxns/logs/error/%x_%A_%a.err
#SBATCH --array=0
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu

script=/home/spn1560/coarse-grain-rxns/scripts/standardize_pickaxe_reactions.py
expansions=(
    1_steps_after_2015_cpds_rules_retrobiocat_rules_w_coreactants_aplusb_True.pk
    # 1_steps_after_2015_cpds_rules_mechinferred_dt_035_rules_before_2015_direct_mcsa_only_w_coreactants_aplusb_True.pk
    # 1_steps_after_2015_cpds_rules_mechinferred_dt_059_rules_before_2015_direct_mcsa_only_w_coreactants_aplusb_True.pk
    # 1_steps_after_2015_cpds_rules_mechinferred_dt_106_rules_before_2015_direct_mcsa_only_w_coreactants_aplusb_True.pk
    # 1_steps_after_2015_cpds_rules_mechinferred_dt_244_rules_before_2015_direct_mcsa_only_w_coreactants_aplusb_True.pk
    # 1_steps_after_2015_cpds_rules_mechinferred_dt_961_rules_before_2015_direct_mcsa_only_w_coreactants_aplusb_True.pk
)
ulimit -c 0
module purge
source /home/spn1560/coarse-grain-rxns/.venv/bin/activate
python $script expansion=${expansions[$SLURM_ARRAY_TASK_ID]}
