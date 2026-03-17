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
#SBATCH --array=0-12
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu

script=/home/spn1560/coarse-grain-rxns/scripts/standardize_pickaxe_reactions.py
expansions=(
    1_steps_after_2015_cpds_rules_imt_rules_before_2015_w_coreactants_aplusb_True.pk
    1_steps_after_2015_cpds_rules_mechinferred_dt_005_rules_before_2015_w_coreactants_aplusb_True.pk
    1_steps_after_2015_cpds_rules_mechinferred_dt_009_rules_before_2015_w_coreactants_aplusb_True.pk
    1_steps_after_2015_cpds_rules_mechinferred_dt_932_rules_before_2015_w_coreactants_aplusb_True.pk
    1_steps_after_2015_cpds_rules_mechinformed_rules_before_2015_w_coreactants_aplusb_True.pk
    1_steps_after_2015_cpds_rules_rc_plus_0_rules_before_2015_w_coreactants_aplusb_True.pk
    1_steps_after_2015_cpds_rules_rc_plus_1_rules_before_2015_w_coreactants_aplusb_True.pk
    1_steps_after_2015_cpds_rules_rc_plus_2_rules_before_2015_w_coreactants_aplusb_True.pk
    1_steps_after_2015_cpds_rules_rc_plus_3_rules_before_2015_w_coreactants_aplusb_True.pk
    1_steps_after_2015_cpds_rules_rc_plus_4_rules_before_2015_w_coreactants_aplusb_True.pk
    1_steps_after_2015_cpds_rules_rdchiral_rules_before_2015_w_coreactants_aplusb_True.pk
    1_steps_after_2015_cpds_rules_mechinferred_dt_021_rules_before_2015_w_coreactants_aplusb_True.pk
    1_steps_after_2015_cpds_rules_mechinferred_dt_069_rules_before_2015_w_coreactants_aplusb_True.pk
)
ulimit -c 0
module purge
source /home/spn1560/coarse-grain-rxns/.venv/bin/activate
python $script expansion=${expansions[$SLURM_ARRAY_TASK_ID]}
