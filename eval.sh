#!/bin/bash
#PBS -l select=1:ncpus=1:gpu_id=0
#PBS -l place=shared
#PBS -o output2604117.txt				
#PBS -e error260417.txt				
#PBS -N nerf
cd ~/graf260108_im64										

source ~/.bashrc											
conda activate graf_gpu	

module load cuda-12.4										
#python3 123.py	
#python3 eval.py configs/carla.yaml --pretrained --rotation_elevation
python eval.py --all --config /Data/home/vicky/graf260108_im64/results/column20260410_gru_disc_Dbn_lrg3_aux/config.yaml --checkpoint /Data/home/vicky/graf260108_im64/results/column20260410_gru_disc_Dbn_lrg3_aux/chkpts/model_00109999.pt