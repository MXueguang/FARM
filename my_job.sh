#!/bin/bash
#SBATCH --mem=64
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:v100l:4
#SBATCH --time=48:0:0
#SBATCH --output=logs/%j.out
export CUDA_AVAILABLE_DEVICES=0,1,2,3

source ~/.bashrc
conda activate farm
python examples/dpr_encoder_projection.py
