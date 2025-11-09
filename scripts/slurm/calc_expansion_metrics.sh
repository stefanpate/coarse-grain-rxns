#!/bin/bash
#SBATCH -A b1039
#SBATCH -p b1039
#SBATCH -N 1
#SBATCH -n 50
#SBATCH --mem=0
#SBATCH -t 14:00:00
#SBATCH --job-name="calc_exp_metrics"
#SBATCH --output=/home/spn1560/coarse-grain-rxns/logs/out/%A
#SBATCH --error=/home/spn1560/coarse-grain-rxns/logs/error/%A
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --array=0-4
#SBATCH --mail-user=stefan.pate@northwestern.edu

# Args
script=/home/spn1560/coarse-grain-rxns/scripts/calc_expansion_metrics.py
processes=50 # Make sure this matches -n above
exp_sweep=(
    # mapped_known_reactions_x_rdchiral_rules.parquet
    # "2_steps_250728_benchmark_starters_rules_imt_rules_w_coreactants_aplusb_True.pk"
    # "2_steps_250728_benchmark_starters_rules_mechinformed_rules_w_coreactants_aplusb_True.pk"
    "2_steps_250728_benchmark_starters_rules_mechinferred_dt_002_rules_w_coreactants_aplusb_True.pk"
    "2_steps_250728_benchmark_starters_rules_mechinferred_dt_006_rules_w_coreactants_aplusb_True.pk"
    "2_steps_250728_benchmark_starters_rules_mechinferred_dt_014_rules_w_coreactants_aplusb_True.pk"
    "2_steps_250728_benchmark_starters_rules_mechinferred_dt_056_rules_w_coreactants_aplusb_True.pk"
    "2_steps_250728_benchmark_starters_rules_mechinferred_dt_956_rules_w_coreactants_aplusb_True.pk"
    # "2_steps_250728_benchmark_starters_rules_rc_plus_1_rules_w_coreactants_aplusb_True.pk"
    # "2_steps_250728_benchmark_starters_rules_rc_plus_2_rules_w_coreactants_aplusb_True.pk"
    # "2_steps_250728_benchmark_starters_rules_rc_plus_3_rules_w_coreactants_aplusb_True.pk"
    # "2_steps_250728_benchmark_starters_rules_rc_plus_4_rules_w_coreactants_aplusb_True.pk"
    # "batch_0_of_20_2_steps_250728_benchmark_starters_rules_rc_plus_0_rules_w_coreactants_aplusb_True.pk"
    # "batch_1_of_20_2_steps_250728_benchmark_starters_rules_rc_plus_0_rules_w_coreactants_aplusb_True.pk"
    # "batch_2_of_20_2_steps_250728_benchmark_starters_rules_rc_plus_0_rules_w_coreactants_aplusb_True.pk"
    # "batch_3_of_20_2_steps_250728_benchmark_starters_rules_rc_plus_0_rules_w_coreactants_aplusb_True.pk"
    # "batch_4_of_20_2_steps_250728_benchmark_starters_rules_rc_plus_0_rules_w_coreactants_aplusb_True.pk"
    # "batch_5_of_20_2_steps_250728_benchmark_starters_rules_rc_plus_0_rules_w_coreactants_aplusb_True.pk"
    # "batch_6_of_20_2_steps_250728_benchmark_starters_rules_rc_plus_0_rules_w_coreactants_aplusb_True.pk"
    # "batch_7_of_20_2_steps_250728_benchmark_starters_rules_rc_plus_0_rules_w_coreactants_aplusb_True.pk"
    # "batch_8_of_20_2_steps_250728_benchmark_starters_rules_rc_plus_0_rules_w_coreactants_aplusb_True.pk"
    # "batch_9_of_20_2_steps_250728_benchmark_starters_rules_rc_plus_0_rules_w_coreactants_aplusb_True.pk"
    # "batch_10_of_20_2_steps_250728_benchmark_starters_rules_rc_plus_0_rules_w_coreactants_aplusb_True.pk"
    # "batch_11_of_20_2_steps_250728_benchmark_starters_rules_rc_plus_0_rules_w_coreactants_aplusb_True.pk"
    # "batch_12_of_20_2_steps_250728_benchmark_starters_rules_rc_plus_0_rules_w_coreactants_aplusb_True.pk"
    # "batch_13_of_20_2_steps_250728_benchmark_starters_rules_rc_plus_0_rules_w_coreactants_aplusb_True.pk"
    # "batch_14_of_20_2_steps_250728_benchmark_starters_rules_rc_plus_0_rules_w_coreactants_aplusb_True.pk"
    # "batch_15_of_20_2_steps_250728_benchmark_starters_rules_rc_plus_0_rules_w_coreactants_aplusb_True.pk"
    # "batch_16_of_20_2_steps_250728_benchmark_starters_rules_rc_plus_0_rules_w_coreactants_aplusb_True.pk"
    # "batch_17_of_20_2_steps_250728_benchmark_starters_rules_rc_plus_0_rules_w_coreactants_aplusb_True.pk"
    # "batch_18_of_20_2_steps_250728_benchmark_starters_rules_rc_plus_0_rules_w_coreactants_aplusb_True.pk"
    # "batch_19_of_20_2_steps_250728_benchmark_starters_rules_rc_plus_0_rules_w_coreactants_aplusb_True.pk"
)

mappings_sweep=(
    # mapped_known_reactions_x_rdchiral_rules.parquet
    # "mapped_known_reactions_x_imt_rules.parquet"
    # "mapped_known_reactions_x_mechinformed_rules.parquet"
    "mapped_known_reactions_x_mechinferred_dt_002_rules.parquet" 
    "mapped_known_reactions_x_mechinferred_dt_006_rules.parquet"
    "mapped_known_reactions_x_mechinferred_dt_014_rules.parquet"
    "mapped_known_reactions_x_mechinferred_dt_056_rules.parquet"
    "mapped_known_reactions_x_mechinferred_dt_956_rules.parquet"
    # "mapped_known_reactions_x_rc_plus_1_rules.parquet"
    # "mapped_known_reactions_x_rc_plus_2_rules.parquet"
    # "mapped_known_reactions_x_rc_plus_3_rules.parquet"
    # "mapped_known_reactions_x_rc_plus_4_rules.parquet"
    # "mapped_known_reactions_x_rc_plus_0_rules.parquet"
    # "mapped_known_reactions_x_rc_plus_0_rules.parquet"
    # "mapped_known_reactions_x_rc_plus_0_rules.parquet"
    # "mapped_known_reactions_x_rc_plus_0_rules.parquet"
    # "mapped_known_reactions_x_rc_plus_0_rules.parquet"
    # "mapped_known_reactions_x_rc_plus_0_rules.parquet"
    # "mapped_known_reactions_x_rc_plus_0_rules.parquet"
    # "mapped_known_reactions_x_rc_plus_0_rules.parquet"
    # "mapped_known_reactions_x_rc_plus_0_rules.parquet"
    # "mapped_known_reactions_x_rc_plus_0_rules.parquet"
    # "mapped_known_reactions_x_rc_plus_0_rules.parquet"
    # "mapped_known_reactions_x_rc_plus_0_rules.parquet"
    # "mapped_known_reactions_x_rc_plus_0_rules.parquet"
    # "mapped_known_reactions_x_rc_plus_0_rules.parquet"
    # "mapped_known_reactions_x_rc_plus_0_rules.parquet"
    # "mapped_known_reactions_x_rc_plus_0_rules.parquet"
    # "mapped_known_reactions_x_rc_plus_0_rules.parquet"
    # "mapped_known_reactions_x_rc_plus_0_rules.parquet"
    # "mapped_known_reactions_x_rc_plus_0_rules.parquet"
    # "mapped_known_reactions_x_rc_plus_0_rules.parquet"
)

# Commands
ulimit -c 0
module purge
source /home/spn1560/coarse-grain-rxns/.venv/bin/activate
python $script mapped_rxns=${mappings_sweep[$SLURM_ARRAY_TASK_ID]} expansion=${exp_sweep[$SLURM_ARRAY_TASK_ID]} processes=$processes