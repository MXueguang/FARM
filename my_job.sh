#!/bin/bash
#SBATCH --mem=32G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:v100l:4
#SBATCH --time=8:0:0
#SBATCH --output=logs/%j.out
export CUDA_AVAILABLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0,1,2,3

source ~/.bashrc
conda activate farm
python examples/dpr_encoder_projection.p