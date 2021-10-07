#!/bin/bash
#
#SBATCH --mail-user=benjamin@tnt.uni-hannover.de
#SBATCH --mail-type=ALL
#SBATCH --partition=cpu_normal
#SBATCH --job-name=meta_box2d_s4_envMetaVehicleRacingEnv_cfargsNone
#SBATCH --output=slurmout/slurm-%j.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32000M



xvfb-run -s "-screen 0 1400x900x24" python run_stablebaselines.py --num_contexts 100 --steps 1000000 --add_context_feature_names_to_logdir --outdir results/singlecontextfeature_0.5_hidecontext/box2d/MetaVehicleRacingEnv  --num_workers 1 --default_sample_std_percentage 0.5 --hide_context --seed 4 --env MetaVehicleRacingEnv --context_feature_args None