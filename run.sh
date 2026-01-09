#!/bin/bash
#PBS -l select=1:ncpus=1:gpu_id=2
#PBS -l place=shared
#PBS -o output260109_07_10.txt				
#PBS -e error260109_07_10.txt				
#PBS -N nerf
cd ~/graf260108_im64										

source ~/.bashrc											
conda activate graftest	

module load cuda-12.4										
#python3 123.py	
#python3 eval.py configs/carla.yaml --pretrained --rotation_elevation
python train.py --config /Data/home/vicky/graf260108_im64/configs/default.yaml