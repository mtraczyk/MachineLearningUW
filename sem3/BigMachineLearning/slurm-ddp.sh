#!/usr/bin/env bash
#
#SBATCH --job-name=bml_lab1
#SBATCH --partition=common
#SBATCH --qos=4gpu1h
#SBATCH --time=5
#SBATCH --output=output.txt
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2

export MASTER_PORT=12340
export WORLD_SIZE=${SLURM_NPROCS}

echo "NODELIST="${SLURM_NODELIST}
echo "WORLD_SIZE="${SLURM_NPROCS}

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

source ~/bml_env/bin/activate

srun python3 ddp_entropy.py