#!/bin/bash
#PBS -l select=1:ncpus=1:gpu_id=2
#PBS -l place=shared
#PBS -o output260223_0730_disinputlabel_MCE_12class_ccsr_tanh_latent_codes_modloss_resb_ccsrpatchloss10_fake_target.txt				
#PBS -e error260223_0730_disinputlabel_MCE_12calss_ccsr_tanh_latent_codes_modloss_resb_ccsrpatchloss10_fake_target.txt				
#PBS -N nerf
cd ~/graf260108_im64										

source ~/.bashrc											
conda activate graftest	

module load cuda-12.4										
#python3 123.py	
#python3 eval.py configs/carla.yaml --pretrained --rotation_elevation
python train.py --config /Data/home/vicky/graf260108_im64/configs/default.yaml