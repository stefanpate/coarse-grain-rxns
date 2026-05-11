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
    ehreact_rules_before_2015_w_coreactants
    ehreact_rules_before_2015_w_coreactants
    ehreact_rules_before_2015_w_coreactants
    # evodex_Bm_rules_before_2015_w_coreactants
    # evodex_Cm_rules_before_2015_w_coreactants
    # evodex_Dm_rules_before_2015_w_coreactants
    # evodex_Em_rules_before_2015_w_coreactants
    # retrobiocat_rules_w_coreactants
)
starters=(
    after_2015_cpds_ds_100
    after_2015_cpds_ds_50
    after_2015_cpds_ds_25
)
generations=1
explicit_h=false
enforce_atom_balance=false
block_inorganic=false
processes=50 # MAKE SURE THIS MATCHES -n above

# Commands
ulimit -c 0
module purge
uv run python $script starters=${starters[$SLURM_ARRAY_TASK_ID]} generations=$generations processes=$processes rules=${rules_sweep[$SLURM_ARRAY_TASK_ID]} explicit_h=$explicit_h enforce_atom_balance=$enforce_atom_balance block_inorganic=$block_inorganic
