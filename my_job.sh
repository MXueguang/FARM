#!/bin/bash
#SBATCH --mem=0
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:v100l:4
#SBATCH --time=48:0:0
export CUDA_AVAILABLE_DEVICES=0,1,2,3

conda activate farm
python examples/dpr_encoder.py

