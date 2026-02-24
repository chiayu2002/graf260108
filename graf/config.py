import numpy as np
import torch
from torchvision.transforms import *
import os

from .datasets import *
from .transforms import FlexGridRaySampler
import yaml
from .utils import polar_to_cartesian, look_at

def to_tensor_and_normalize(x):
        return x * 2 - 1

def get_data(config):
    H = W = imsize = config['data']['imsize']
    dset_type = config['data']['type']
    fov = config['data']['fov']

    transforms = Compose([
        Resize(imsize), #調整輸入圖片的大小
        ToTensor(), #把圖片轉換成pytorch可以處理的格式，並把像素值從[0,255]規一化成[0,1]
        Lambda(to_tensor_and_normalize), #把值從[0,1]轉換成[-1,1]
    ])

    kwargs = {
        'data_dirs': config['data']['datadir'],
        'transforms': transforms
    }

    if dset_type == 'carla':
        dset = Carla(**kwargs)
    
    elif dset_type == 'RS307_0_i2':
        dset = RS307_0_i2(**kwargs)

    dset.H = dset.W = imsize
    dset.focal = W/2 * 1 / np.tan((.5 * fov * np.pi/180.))
    radius = config['data']['radius']
    if isinstance(radius, str):
        radius = tuple(float(r) for r in radius.split(','))
    dset.radius = radius



    print('Loaded {}'.format(dset_type), imsize, len(dset), [H,W,dset.focal,dset.radius], config['data']['datadir'])
    return dset, [H,W,dset.focal,dset.radius]

def get_render_poses(radius, angle_range=(0, 360), theta=0, N=40, swap_angles=False):   #用在eval的時候
    poses = []
    theta = max(0.1, theta)
    for angle in np.linspace(angle_range[0],angle_range[1],N+1)[:-1]:
        angle = max(0.1, angle)
        if swap_angles:
            loc = polar_to_cartesian(radius, theta, angle, deg=True)
        else:
            loc = polar_to_cartesian(radius, angle, theta, deg=True)
        R = look_at(loc)[0]
        RT = np.concatenate([R, loc.reshape(3, 1)], axis=1)
        poses.append(RT)
    return torch.from_numpy(np.stack(poses))



def build_models(config, disc=True):
    from argparse import Namespace
    from submodules.nerf_pytorch.run_nerf_mod import create_nerf
    from .models.generator import Generator
    from .models.discriminator import Discriminator #, QHead, DHead

    config_nerf = Namespace(**config['nerf'])
    # Update config for NERF
    config_nerf.chunk = min(config['training']['chunk'], 1024*config['training']['batch_size'])     # let batch size for training with patches limit the maximal memory
    config_nerf.netchunk = config['training']['netchunk']
    config_nerf.feat_dim = config['z_dist']['dim']
    config_nerf.num_class = config['nerf']['num_classes']
    # config_nerf.feat_dim_appearance = config['z_dist']['dim_appearance']

    render_kwargs_train, render_kwargs_test, params, named_parameters = create_nerf(config_nerf)

    bds_dict = {'near': config['data']['near'], 'far': config['data']['far']}
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    ray_sampler = FlexGridRaySampler(N_samples=config['ray_sampler']['N_samples'],
                                     min_scale=config['ray_sampler']['min_scale'],
                                     max_scale=config['ray_sampler']['max_scale'],
                                     scale_anneal=config['ray_sampler']['scale_anneal'],
                                     orthographic=config['data']['orthographic'],
                                    #  random_shift = False,
                                    #  random_scale = False
                                     )

    H, W, f, r = config['data']['hwfr']

    generator = Generator(H, W, f, r,
                          ray_sampler=ray_sampler,
                          render_kwargs_train=render_kwargs_train, render_kwargs_test=render_kwargs_test,
                          parameters=params, named_parameters=named_parameters,
                          chunk=config_nerf.chunk,
                          range_u=(float(config['data']['umin']), float(config['data']['umax'])),
                          range_v=(float(config['data']['vmin']), float(config['data']['vmax'])),
                          orthographic=config['data']['orthographic'],
                          v=config['data']['v'],
                          use_default_rays=config['data']['use_default_rays'],
                          use_ccsr=config['ccsr']['enabled'],  # 啟用CCSR
                          num_views=8
                          )
    
    discriminator = None
    if disc:
        disc_kwargs = {'nc': 3,       # channels for patch discriminator
                       'ndf': config['discriminator']['ndf'],
                       'imsize': 64,  #int(np.sqrt(config['ray_sampler']['N_samples'])),
                       'hflip': config['discriminator']['hflip'],
                       'num_classes':config['discriminator']['num_classes']
                        }

        discriminator = Discriminator(**disc_kwargs)

    # qhead = QHead()
    # dhead = DHead()
    return generator, discriminator   #, qhead, dhead

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(outpath, config):
    from yaml import safe_dump
    with open(outpath, 'w') as f:
        safe_dump(config, f)

def build_lr_scheduler(optimizer, config, last_epoch=-1):
    import torch.optim as optim
    step_size = config['training']['lr_anneal_every']
    if isinstance(step_size, str):
        milestones = [int(m) for m in step_size.split(',')]
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=config['training']['lr_anneal'],
            last_epoch=last_epoch)
    else:
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=config['training']['lr_anneal'],
            last_epoch=last_epoch
        )
    return lr_scheduler