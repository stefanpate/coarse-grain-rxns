#!/bin/bash
#SBATCH -A b1039
#SBATCH -p b1039
#SBATCH -N 1
#SBATCH -n 50
#SBATCH --mem=80GB
#SBATCH -t 48:00:00
#SBATCH --job-name="expstd"
#SBATCH --output=/home/spn1560/coarse-grain-rxns/logs/out/%x_%A_%a.out
#SBATCH --error=/home/spn1560/coarse-grain-rxns/logs/error/%x_%A_%a.err
#SBATCH --array=0-3
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu

# Args
script=/home/spn1560/coarse-grain-rxns/scripts/expand.py
script2=/home/spn1560/coarse-grain-rxns/scripts/standardize_pickaxe_reactions.py
expansions=(
    1_steps_after_2015_cpds_rules_evodex_Bm_rules_before_2015_w_coreactants_aplusb_True.pk
    1_steps_after_2015_cpds_rules_evodex_Cm_rules_before_2015_w_coreactants_aplusb_True.pk
    1_steps_after_2015_cpds_rules_evodex_Dm_rules_before_2015_w_coreactants_aplusb_True.pk
    1_steps_after_2015_cpds_rules_evodex_Em_rules_before_2015_w_coreactants_aplusb_True.pk
)
rules_sweep=(
    evodex_Bm_rules_before_2015_w_coreactants
    evodex_Cm_rules_before_2015_w_coreactants
    evodex_Dm_rules_before_2015_w_coreactants
    evodex_Em_rules_before_2015_w_coreactants
)
starters=after_2015_cpds
generations=1
explicit_h=true
processes=50 # MAKE SURE THIS MATCHES -n above

# Commands
ulimit -c 0
module purge
uv run python $script starters=$starters generations=$generations processes=$processes rules=${rules_sweep[$SLURM_ARRAY_TASK_ID]} explicit_h=$explicit_h
uv run python $script2 expansion=${expansions[$SLURM_ARRAY_TASK_ID]}
