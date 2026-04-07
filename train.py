import argparse
import os, importlib, json, glob
import time
import torch.nn.functional as F
import torch.nn as nn
import random
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import wandb
import pprint
import sys
sys.path.append('submodules')
import torchvision.utils as vutils

from graf.gan_training import Evaluator
from graf.config import get_data, build_models, load_config, save_config, build_lr_scheduler
from graf.utils import get_zdist
from graf.train_step import compute_grad2, compute_loss, toggle_grad
from graf.transforms import ImgToPatch

from GAN_stability.gan_training.checkpoints_mod import CheckpointIO


def setup_directories(config):
    out_dir = os.path.join(config['training']['outdir'], config['expname'])
    checkpoint_dir = os.path.join(out_dir, 'chkpts')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    return out_dir, checkpoint_dir

def initialize_training(config, device):
    extractor_path = config['data']['extractor_path']
    extractor_args = json.load(open(glob.glob("/Data/home/vicky/graf260108_im64/HystereticGRU/2026-03-24_17-13-01/args.json", recursive=True)[0], "r"))
    extractor_args = argparse.Namespace(**extractor_args)
    extractor = importlib.import_module(f"graf.models.HystereticPrediction").__dict__[extractor_args.architecture](**vars(extractor_args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = extractor.to(device)
    state_dict = torch.load(glob.glob(extractor_path, recursive=True)[0])["state_dict"]
    status = extractor.load_state_dict(state_dict)
    print("Extractor Loading Status: ", status)

    train_dataset, hwfr = get_data(config, extractor, extractor_args)

    if config['data']['orthographic']:
        hw_ortho = (config['data']['far']-config['data']['near'],) * 2
        hwfr[2] = hw_ortho
    config['data']['hwfr'] = hwfr
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['nworkers'],
        shuffle=True, 
        pin_memory=True,
        sampler=None, 
        drop_last=True,
        generator=torch.Generator(device='cuda:0')
    )
    
    generator, discriminator = build_models(config)
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    return train_loader, train_dataset, generator, discriminator

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_random_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()

    config = load_config(args.config)
    config['data']['fov'] = float(config['data']['fov'])
    restart_every = config['training']['restart_every']
    batch_size = config['training']['batch_size']
    fid_every = config['training']['fid_every']
    save_best = config['training']['save_best']
    reg_param = config['training']['reg_param']
    device = torch.device("cuda:0")
    
    out_dir, checkpoint_dir = setup_directories(config)
    save_config(os.path.join(out_dir, 'config.yaml'), config)
    
    wandb.init(
        project=config['wandb']['project'],
        name=config['wandb']['name'],
        config=config
    )

    train_loader, train_dataset, generator, discriminator = initialize_training(config, device)

    file_path = os.path.join(out_dir, "model_architecture.txt")
    with open(file_path, 'w') as f:
        f.write('Discriminator Architecture:\n')
        f.write('-' * 50 + '\n')
        f.write(str(discriminator))
        f.write('\n\n')
        f.write('Generator Architecture:\n')
        f.write('-' * 50 + '\n')
        pprint.pprint(generator.module_dict, stream=f)
    wandb.save(file_path)

    lr_g = config['training']['lr_g']
    lr_d = config['training']['lr_d']
    g_params = generator.parameters()
    d_params = discriminator.parameters()
    g_optimizer = optim.RMSprop(g_params, lr=lr_g, alpha=0.99, eps=1e-8)
    d_optimizer = optim.RMSprop(d_params, lr=lr_d, alpha=0.99, eps=1e-8)

    hwfr = config['data']['hwfr']
    img_to_patch = ImgToPatch(generator.ray_sampler, hwfr[:3])
    
    checkpoint_io = CheckpointIO(checkpoint_dir=checkpoint_dir)
    checkpoint_io.register_modules(
        discriminator=discriminator,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        **generator.module_dict
    )
    
    zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'], device=device)

    evaluator = Evaluator(fid_every > 0, generator, zdist, None,
                          batch_size=batch_size, device=device, inception_nsamples=33)
    val_loader = train_loader
    if fid_every > 0:
        fid_cache_file = os.path.join(out_dir, 'fid_cache_train.npz')
        kid_cache_file = os.path.join(out_dir, 'kid_cache_train.npz')
        evaluator.inception_eval.initialize_target(val_loader, cache_file=fid_cache_file, act_cache_file=kid_cache_file)
    
    # 預先快取各實驗的 hidden_state
    cached_hidden_states = {}
    for exp_name, hs in train_dataset.hidden_state.items():
        cached_hidden_states[exp_name] = hs.to(device)
    print(f"Cached hidden states for: {list(cached_hidden_states.keys())}")

    fid_best = float('inf')
    kid_best = float('inf')
    
    it = epoch_idx = -1
    tstart = t0 = time.time()
    
    g_scheduler = build_lr_scheduler(g_optimizer, config, last_epoch=it)
    d_scheduler = build_lr_scheduler(d_optimizer, config, last_epoch=it)

    # ========================================================
    # [變更] 移除了:
    # - MCE_Loss / criterion_cls（不再做分類）
    # - n_each_task（不再需要）
    # - lambda_cls_d / lambda_cls_g（不再需要）
    # - CCSRLoss（如果你也不需要 CCSR 的話）
    #
    # Discriminator 現在只做真假判別，
    # 條件資訊完全透過 hidden_state 注入
    # ========================================================

    while True:
        epoch_idx += 1
        for x_real, label, hidden_state in tqdm(train_loader, desc=f"Epoch {epoch_idx}"):
            it += 1

            x_real = x_real.to(device)
            label = label.to(device)
            hidden_state = hidden_state.to(device)

            generator.ray_sampler.iterations = it
            toggle_grad(generator, False)
            toggle_grad(discriminator, True)
            generator.train()
            discriminator.train()

            # ========== Discriminator Step ==========
            d_optimizer.zero_grad()

            rgbs = img_to_patch(x_real)
            rgbs.requires_grad_(True)

            z = zdist.sample((batch_size,))
            
            # ========================================================
            # [變更] discriminator 改為接收 hidden_state 而非 label
            # 原本: d_real, label_pred = discriminator(rgbs, label)
            # 現在: d_real = discriminator(rgbs, hidden_state)
            # 回傳值只有真假判別分數，不再有分類預測
            # ========================================================
            
            # Real
            d_real = discriminator(rgbs, hidden_state)
            dloss_real = compute_loss(d_real, 1)
            reg = reg_param * compute_grad2(d_real, rgbs).mean()
            
            # Fake
            with torch.no_grad():
                x_fake, _ = generator(z, label, hidden_state)
            x_fake.requires_grad_()
            
            d_fake = discriminator(x_fake, hidden_state)
            dloss_fake = compute_loss(d_fake, 0)

            total_d_loss = dloss_real + dloss_fake + reg
            total_d_loss.backward()
            d_optimizer.step()
            d_scheduler.step()

            # ========== Generator Step ==========
            if config['nerf']['decrease_noise']:
                generator.decrease_nerf_noise(it)

            toggle_grad(generator, True)
            toggle_grad(discriminator, False)
            generator.train()
            discriminator.train()
            g_optimizer.zero_grad()

            z = zdist.sample((batch_size,))
            x_fake, _ = generator(z, label, hidden_state)
            
            d_fake = discriminator(x_fake, hidden_state)
            gloss = compute_loss(d_fake, 1)

            gloss.backward()
            g_optimizer.step()
            g_scheduler.step()

            current_lr_g = g_optimizer.param_groups[0]['lr']
            current_lr_d = d_optimizer.param_groups[0]['lr']
            
            if (it + 1) % config['training']['print_every'] == 0:
                wandb.log({
                    "loss/generator": gloss,
                    "loss/discriminator": total_d_loss,
                    "loss/dloss_real": dloss_real,
                    "loss/dloss_fake": dloss_fake,
                    "loss/regularizer": reg,
                    "learning rate/generator": current_lr_g,
                    "learning rate/discriminator": current_lr_d,
                    "iteration": it
                })

            # Sample
            if ((it % config['training']['sample_every']) == 0) or ((it < 500) and (it % 100 == 0)):
                plist = []
                angle_positions = [(i/8, 0.5) for i in range(8)] 
                ztest = zdist.sample((batch_size,))

                vec_307 = [1.0, 0.0,  1.0, 0.0, 0.0,  0.0, 1.0]

                for i, (u, v) in enumerate(angle_positions):
                    poses = generator.sample_select_pose(u, v)
                    plist.append(poses)
                ptest = torch.stack(plist)

                angles = [0, 45, 90, 135, 180, 225, 270, 315]
                test_labels_list = []
                for angle in angles:
                    lbl = vec_307 + [0.5, float(angle)]
                    test_labels_list.append(lbl)
                label_test_all = torch.tensor(test_labels_list, dtype=torch.float32).to(device)

                hs_307 = cached_hidden_states['RS307']
                hs_test = hs_307.unsqueeze(0).expand(8, -1)

                rgb, depth, acc = evaluator.create_samples(ztest.to(device), label_test_all, hs_test, ptest)

                grid_rgb = vutils.make_grid(rgb.detach().cpu(), nrow=8, normalize=True)
                grid_depth = vutils.make_grid(depth.detach().cpu(), nrow=8, normalize=True)
                grid_acc = vutils.make_grid(acc.detach().cpu(), nrow=8, normalize=True)
                
                wandb.log({
                    "sample/rgb": [wandb.Image(grid_rgb, caption=f"RGB at iter {it}")],
                    "sample/depth": [wandb.Image(grid_depth, caption=f"Depth at iter {it}")],
                    "sample/acc": [wandb.Image(grid_acc, caption=f"Acc at iter {it}")],
                    "epoch_idx": epoch_idx,
                    "iteration": it
                })

            # FID/KID
            if fid_every > 0 and ((it + 1) % fid_every) == 0:
                fid, kid = evaluator.compute_fid_kid(label, hidden_state)
                wandb.log({
                    "validation/fid": fid,
                    "validation/kid": kid,
                    "iteration": it
                })
                torch.cuda.empty_cache()
                if save_best == 'fid' and fid < fid_best:
                    fid_best = fid
                    print('Saving best model based on FID...')
                    wandb.run.summary["best_fid"] = fid_best
                    checkpoint_io.save('model_best.pt', it=it, epoch_idx=epoch_idx, fid_best=fid_best, kid_best=kid_best, save_to_wandb=True)
                    torch.cuda.empty_cache()
                elif save_best == 'kid' and kid < kid_best:
                    kid_best = kid
                    print('Saving best model based on KID...')
                    wandb.run.summary["best_kid"] = kid_best
                    checkpoint_io.save('model_best.pt', it=it, epoch_idx=epoch_idx, fid_best=fid_best, kid_best=kid_best, save_to_wandb=True)
                    torch.cuda.empty_cache()

            if ((it + 1) % 10000) == 0:
                print('Saving backup...')
                checkpoint_io.save('model_%08d.pt' % it, it=it, epoch_idx=epoch_idx, save_to_wandb=True)

            if time.time() - t0 > config['training']['save_every']:
                checkpoint_io.save(
                    config['training']['model_file'], 
                    it=it, 
                    epoch_idx=epoch_idx,
                    save_to_wandb=True
                )
                t0 = time.time()
                
                if (restart_every > 0 and t0 - tstart > restart_every):
                    return

if __name__ == '__main__':
    main()