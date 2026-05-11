#!/bin/bash
#PBS -l select=1:ncpus=1:gpu_id=3
#PBS -l place=shared
#PBS -o output260510_twostage_ttur_fullimg64_sr256.txt				
#PBS -e error20260510_twostage_ttur_fullimg64_sr256.txt				
#PBS -N nerf
cd ~/graf260108_im64										

source ~/.bashrc											
conda activate graf_gpu	

module load cuda-12.4										
python train_twostage.py --config /Data/home/vicky/graf260108_im64/configs/twostage.yaml