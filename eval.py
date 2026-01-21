import argparse
import os
from os import path
import numpy as np
import time
import copy
import csv
import random
import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')
from torchvision.utils import save_image

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib
matplotlib.use('Agg')


import sys
sys.path.append('submodules')        # needed to make imports work in GAN_stability

from submodules.GAN_stability.gan_training.checkpoints_mod import CheckpointIO
from graf.gan_training import Evaluator as Evaluator
from graf.config import get_data, build_models, get_render_poses, load_config
from graf.utils import count_trainable_parameters, to_phi, to_theta, get_nsamples, get_zdist
from graf.transforms import ImgToPatch

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

checkpoint_path = '/Data/home/vicky/graf260108_im64/results/column260119_0730_reg10_acgan_real_target_modnerf_inputlabel10_7class_skip3_BCE_lr3/chkpts/model_00049999.pt'

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(
        description='Train a GAN with different regularization strategies.'
    )
    parser.add_argument('--config', default='/Data/home/vicky/graf260108_im64/results/column260119_0730_reg10_acgan_real_target_modnerf_inputlabel10_7class_skip3_BCE_lr3/config.yaml', type=str, help='Path to config file.')
    parser.add_argument('--fid_kid', action='store_true', help='Evaluate FID and KID.')
    parser.add_argument('--create_sample', default= True, help='Generate videos with changing camera pose.')
    parser.add_argument('--rotation_elevation', action='store_true', help='Generate videos with changing camera pose.')
    parser.add_argument('--shape_appearance', action='store_true', help='Create grid image showing shape/appearance variation.')
    parser.add_argument('--pretrained', action='store_true', help='Load pretrained model.')

    args = parser.parse_args()
    config = load_config(args.config)
    config['data']['fov'] = float(config['data']['fov'])

    # Short hands
    batch_size = config['training']['batch_size']
    out_dir = os.path.join(config['training']['outdir'], config['expname'])
    checkpoint_dir = path.join(out_dir, 'chkpts')
    eval_dir = os.path.join(out_dir, 'eval')
    os.makedirs(eval_dir, exist_ok=True)
    fid_kid = int(args.fid_kid)

    config['training']['nworkers'] = 0

    def set_random_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多 GPU
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True  # 確定性算法
        torch.backends.cudnn.benchmark = False  # 關閉自動優化

    set_random_seed(0)

    # Logger
    checkpoint_io = CheckpointIO(
        checkpoint_dir=checkpoint_dir
    )

    device = torch.device("cuda:0")

    # Dataset
    train_dataset, hwfr = get_data(config)

    config['data']['hwfr'] = hwfr         # add for building generator
    print(train_dataset, hwfr)
    
    val_dataset = train_dataset                 # evaluate on training dataset for GANs
    if args.fid_kid:
        val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                num_workers=config['training']['nworkers'],
                shuffle=True, pin_memory=False, sampler=None, drop_last=False,   # enable shuffle for fid/kid computation
                generator=torch.Generator(device='cuda:0')
        )

    # Create models
    generator, _ = build_models(config, disc=False)
    print('Generator params: %d' % count_trainable_parameters(generator))

    # Put models on gpu if needed
    generator = generator.to(device)

    # input transform
    img_to_patch = ImgToPatch(generator.ray_sampler, hwfr[:3])

    # Register modules to checkpoint
    checkpoint_io.register_modules(
        **generator.module_dict  # treat NeRF specially
    )

    # Distributions
    zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'],
                      device=device)
    
    # Evaluator
    evaluator = Evaluator(fid_kid, generator, zdist, None,
                          batch_size=batch_size, device=device)

    # Train
    tstart = t0 = time.time()
    
    # Load checkpoint
    load_dict = checkpoint_io.load(checkpoint_path)
    it = load_dict.get('it', -1)
    epoch_idx = load_dict.get('epoch_idx', -1)

    def create_labels(num_samples, label_value):
        return torch.full((num_samples, 1), label_value)
    
    if args.create_sample:
        N_samples = 8
        N_poses = 20            # corresponds to number of frames
        render_radius = config['data']['radius']
        if isinstance(render_radius, str):  # use maximum radius
            render_radius = float(render_radius.split(',')[1])

        # original_positions = [(0.15, 0.24),(0.29, 0.24), (0.56, 0.24), (0.71, 0.24), (0.82, 0.47), 
        #   (0.82, 0.38), (0.82, 0.22),(0.82, 0.14)]
        # angle_positions = [(u+0.75 , v) for u, v in original_positions] 
        angle_positions= [(i/8+0.25, 0.5) for i in range(8)] 
        # label = create_labels(N_samples, 1)

        z = zdist.sample((N_samples,))

        vec_307 = [1.0, 0.0,  1.0, 0.0, 0.0,  0.0, 1.0,  0.5, 0.0]
        vec_330 = [1.0, 0.0,  0.0, 0.0, 1.0,  1.0, 0.0,  0.5, 0.0]
        # test_labels_list = []
        # test_labels_list.append(vec_307)
        test_labels_tensor = torch.tensor([vec_307] * N_samples, dtype=torch.float32).to(device)

        all_rgb = []
        for i, (u, v) in enumerate(angle_positions):
            # position_angle = (azimuth + 180) % 360
            print(f"處理角度位置 {i}: ({u}, {v})")
            poses = generator.sample_select_pose(u ,v)
            # print(poses)
            rgb, depth, acc = evaluator.create_samples(z.to(device)[i:i+1], test_labels_tensor[i:i+1], poses.unsqueeze(0))
            all_rgb.append(rgb)

        rgb = torch.cat(all_rgb, dim=0)
        rgb = ((rgb / 2 + 0.5).clamp(0, 1) * 255).to(torch.uint8)
        rgb = rgb.float() / 255        
        n_vis = 8
        filename = 'fake_samples_307.png'
        outpath = os.path.join(eval_dir, filename)
        save_image(rgb, outpath, nrow=n_vis)

    if args.rotation_elevation:
        N_samples = 1
        N_poses = 180            # corresponds to number of frames
        render_radius = config['data']['radius']
        if isinstance(render_radius, str):  # use maximum radius
            render_radius = float(render_radius.split(',')[1])

        # compute render poses
        def get_render_poses_rotation_elevation(N_poses=float('inf')):
            """Compute equidistant render poses varying azimuth and polar angle, respectively."""
            range_theta = to_theta(0.5)
            range_phi = (to_phi(config['data']['umin']+0.5), to_phi(config['data']['umax']+0.5))

            N_phi = min(int(range_phi[1] - range_phi[0]), N_poses)  # at least 1 frame per degree

            render_poses_rotation = get_render_poses(render_radius, angle_range=range_phi, theta=range_theta, N=N_phi)


            return {'rotation': render_poses_rotation}

        z = zdist.sample((N_samples,))
        label = create_labels(N_samples, 1)

        for name, poses in get_render_poses_rotation_elevation(N_poses).items():
            outpath = os.path.join(eval_dir, '{}/'.format(name))
            os.makedirs(outpath, exist_ok=True)
            evaluator.make_video(outpath, z, label, poses, as_gif=False)
            torch.cuda.empty_cache()
    # Evaluation loop
    if args.fid_kid:
        # Specifically generate samples that can be saved
        n_samples = 1000
        ztest = zdist.sample((n_samples,))
        samples, _, _ = evaluator.create_samples(ztest.to(device), label.to(device))
        samples = (samples / 2 + 0.5).mul_(255).clamp_(0, 255).to(torch.uint8)      # convert to unit8

        filename = 'samples_fid_kid_{:06d}.npy'.format(n_samples)
        outpath = os.path.join(eval_dir, filename)
        np.save(outpath, samples.numpy())
        print('Saved {} samples to {}.'.format(n_samples, outpath))

        samples = samples.to(torch.float) / 255

        n_vis = 8
        filename = 'fake_samples.png'
        outpath = os.path.join(eval_dir, filename)
        save_image(samples[:n_vis**2].clone(), outpath, nrow=n_vis)
        print('Plot examples under {}.'.format(outpath))

        filename = 'real_samples.png'
        outpath = os.path.join(eval_dir, filename)
        sample, _ = get_nsamples(val_loader, n_vis**2)
        real = sample / 2 + 0.5
        save_image(real[:n_vis ** 2].clone(), outpath, nrow=n_vis)
        print('Plot examples under {}.'.format(outpath))

        # Compute FID and KID
        fid_cache_file = os.path.join(out_dir, 'fid_cache_train.npz')
        kid_cache_file = os.path.join(out_dir, 'kid_cache_train.npz')
        evaluator.inception_eval.initialize_target(val_loader, cache_file=fid_cache_file, act_cache_file=kid_cache_file)

        samples = samples * 2 - 1
        sample_loader = torch.utils.data.DataLoader(
            samples,
            batch_size=evaluator.batch_size, num_workers=config['training']['nworkers'],
            shuffle=False, pin_memory=False, sampler=None, drop_last=False,
            generator=torch.Generator(device='cuda:0')
        )
        fid, kid = evaluator.compute_fid_kid(label, sample_loader)

        filename = 'fid_kid.csv'
        outpath = os.path.join(eval_dir, filename)
        with open(outpath, mode='w') as csv_file:
            fieldnames = ['fid', 'kid']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerow({'fid': fid, 'kid': kid})

        print('Saved FID ({:.1f}) and KIDx100 ({:.2f}) to {}.'.format(fid, kid*100, outpath))