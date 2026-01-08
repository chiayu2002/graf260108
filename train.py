import argparse
import os
import time
import torch.nn.functional as F
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

from graf.gan_training import Evaluator
from graf.config import get_data, build_models, load_config, save_config, build_lr_scheduler
from graf.utils import get_zdist
from graf.train_step import compute_grad2, compute_loss, save_data, wgan_gp_reg, toggle_grad, MCE_Loss, CCSRNeRFLoss
from graf.transforms import ImgToPatch
# from graf.models.vit_model import ViewConsistencyTransformer
 
from GAN_stability.gan_training.checkpoints_mod import CheckpointIO

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def setup_directories(config):
    out_dir = os.path.join(config['training']['outdir'], config['expname'])
    checkpoint_dir = os.path.join(out_dir, 'chkpts')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    return out_dir, checkpoint_dir

def initialize_training(config, device):
    # dataset
    train_dataset, hwfr= get_data(config)
    if config['data']['orthographic']:
        hw_ortho = (config['data']['far']-config['data']['near'],) * 2
        hwfr[2] = hw_ortho
    config['data']['hwfr'] = hwfr
    
    # train_loader
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
    
    # Create models
    generator, discriminator = build_models(config) #, qhead, dhead
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    # qhead = qhead.to(device)
    # dhead = dhead.to(device)
    
    return train_loader, generator, discriminator #, qhead, dhead

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多 GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # 確定性算法
    torch.backends.cudnn.benchmark = False  # 關閉自動優化

def main():
    set_random_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()

    # load config
    config = load_config(args.config)
    config['data']['fov'] = float(config['data']['fov'])
    restart_every = config['training']['restart_every']
    batch_size=config['training']['batch_size']
    fid_every = config['training']['fid_every']
    save_best = config['training']['save_best']
    device = torch.device("cuda:0")
    
    # 創建目錄
    out_dir, checkpoint_dir = setup_directories(config)
    save_config(os.path.join(out_dir, 'config.yaml'), config)
    
    # 初始化 wandb
    wandb.init(
        project="graf250520",
        # entity="vicky20020808",
        name="RS307 330",
        config=config
    )

    # 初始化model
    train_loader, generator, discriminator = initialize_training(config, device) #, qhead, dhead 

    ccsr_nerf_loss = CCSRNeRFLoss().to(device)

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

    # 優化器
    lr_g = config['training']['lr_g']
    lr_d = config['training']['lr_d']
    g_params = generator.parameters()
    d_params = discriminator.parameters()
    # g_params = list(generator.parameters()) + list(qhead.parameters())
    # d_params = list(discriminator.parameters()) + list(dhead.parameters())
    g_optimizer = optim.RMSprop(g_params, lr=lr_g, alpha=0.99, eps=1e-8)
    d_optimizer = optim.RMSprop(d_params, lr=lr_d, alpha=0.99, eps=1e-8)
    # g_optimizer = optim.Adam(g_params, lr=lr_g, betas=(0.5, 0.999), eps=1e-8)
    # d_optimizer = optim.Adam(d_params, lr=lr_d, betas=(0.5, 0.999), eps=1e-8)

    #get patch
    hwfr = config['data']['hwfr']
    img_to_patch = ImgToPatch(generator.ray_sampler, hwfr[:3])
    
    # 設置檢查點
    checkpoint_io = CheckpointIO(checkpoint_dir=checkpoint_dir)
    checkpoint_io.register_modules(
        discriminator=discriminator,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        **generator.module_dict
    )
    
    zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'], device=device)

    # Evaluator
    evaluator = Evaluator(fid_every > 0, generator, zdist, None,
                          batch_size=batch_size, device=device, inception_nsamples=33)
    val_loader = train_loader
    if fid_every > 0:
        fid_cache_file = os.path.join(out_dir, 'fid_cache_train.npz')
        kid_cache_file = os.path.join(out_dir, 'kid_cache_train.npz')
        evaluator.inception_eval.initialize_target(val_loader, cache_file=fid_cache_file, act_cache_file=kid_cache_file)
    
    fid_best = float('inf')
    kid_best = float('inf')
    
    it = epoch_idx = -1
    tstart = t0 = time.time()
    
    g_scheduler = build_lr_scheduler(g_optimizer, config, last_epoch=it)
    d_scheduler = build_lr_scheduler(d_optimizer, config, last_epoch=it)
    config['training']['lr_g'] = lr_g
    config['training']['lr_d'] = lr_d

    while True:
        epoch_idx += 1
        for x_real, label in tqdm(train_loader, desc=f"Epoch {epoch_idx}"):
            it += 1

            mce_loss = MCE_Loss()
            first_label = label[:,0]
            first_label = first_label.long()
            batch_size = first_label.size(0)
            one_hot = torch.zeros(batch_size, 1, device=first_label.device)
            one_hot.scatter_(1, first_label.unsqueeze(1), 1)
            
            generator.ray_sampler.iterations = it
            toggle_grad(generator, False)
            toggle_grad(discriminator, True)
            generator.train()
            discriminator.train()
            # qhead.train()
            # dhead.train()

            # Discriminator updates
            d_optimizer.zero_grad()

            x_real = x_real.to(device)
            rgbs = img_to_patch(x_real)
            rgbs.requires_grad_(True)

            z = zdist.sample((batch_size,))
            
            #real data
            d_real, label_real = discriminator(rgbs, label)
            # output1 = discriminator(rgbs, label)
            # d_real = dhead(output1)
            dloss_real = compute_loss(d_real, 1)
            one_hot = one_hot.to(label_real.device)
            # d_label_loss = mce_loss([2], label_real, one_hot)
            # dloss_real.backward()
            # dloss_real.backward(retain_graph=True)
            reg = 80. * compute_grad2(d_real, rgbs).mean()
            # reg.backward()
            
            #fake data
            with torch.no_grad():
                x_fake, _ = generator(z, label)
            x_fake.requires_grad_()

            d_fake, _ = discriminator(x_fake, label)
            # output2 = discriminator(x_fake, label)
            # d_fake = dhead(output2)
            dloss_fake = compute_loss(d_fake, 0)
            # dloss_fake.backward()
            # reg = 10. * wgan_gp_reg(discriminator, rgbs, x_fake, label)
            # reg.backward()

            # dloss = dloss_real + dloss_fake
            total_d_loss = dloss_real + dloss_fake + reg #+ d_label_loss 
            # dloss_all = dloss_real + dloss_fake +reg
            total_d_loss.backward()
            d_optimizer.step()
            d_scheduler.step()

            # Generators updates
            if config['nerf']['decrease_noise']:
                generator.decrease_nerf_noise(it)

            toggle_grad(generator, True)
            toggle_grad(discriminator, False)
            generator.train()
            discriminator.train()
            # qhead.train()
            # dhead.train()
            g_optimizer.zero_grad()

            z = zdist.sample((batch_size,))
            # x_fake, _= generator(z, label)
            x_fake, _, ccsr_output = generator(z, label, return_ccsr_output=True)
            d_fake, label_fake = discriminator(x_fake, label)
            # output = discriminator(x_fake, label)
            # d_fake = dhead(output)
            # g_label_loss = mce_loss([2], label_fake, one_hot)

            gloss = compute_loss(d_fake, 1) 
            ccsr_consistency_loss = ccsr_nerf_loss(ccsr_output, x_fake)
            # label_fake = qhead(output)
            # label_loss = mce_loss([2], label_fake, one_hot.to(device))
            gloss_all = gloss + ccsr_consistency_loss#+ g_label_loss

            gloss_all.backward()
            # gloss.backward()
            g_optimizer.step()
            g_scheduler.step()

            current_lr_g = g_optimizer.param_groups[0]['lr']
            current_lr_d = d_optimizer.param_groups[0]['lr']
            # wandb
            if (it + 1) % config['training']['print_every'] == 0:
                wandb.log({
                    "loss/generator": gloss,
                    "loss/ccsr_consistency": ccsr_consistency_loss,
                    "loss/generator_total": gloss_all,
                    "loss/discriminator": total_d_loss,
                    "loss/regularizer": reg,
                    "learning rate/generator": current_lr_g,
                    "learning rate/discriminator": current_lr_d,
                    "iteration": it
                })
            
            # 在需要儲存資料的位置，例如在訓練迴圈中特定迭代次數時
            # if (it % 6000 == 0):
            #     # 在這裡使用當前的 label 和 rays
            #     save_data(label, rays, it, save_dir=os.path.join(out_dir, 'saved_data'))

            # (ii) Sample if necessary
            if ((it % config['training']['sample_every']) == 0) or ((it < 500) and (it % 100 == 0)):
                # is_training = generator.use_test_kwargs
                # generator.eval()  
                plist = []
                angle_positions = [(i/8, 0.5) for i in range(8)] 
                ztest = zdist.sample((batch_size,))
                label_test = torch.tensor([[0] if i < 4 else [0] for i in range(batch_size)])

                # save_dir = os.path.join(out_dir, 'poses')
                # os.makedirs(save_dir, exist_ok=True)

                for i, (u, v) in enumerate(angle_positions):
                    # print(f"指定角度:{u}, 轉換後角度:{position_angle}")
                    poses = generator.sample_select_pose(u ,v)
                    plist.append(poses)
                ptest = torch.stack(plist)

                rgb, depth, acc = evaluator.create_samples(ztest.to(device), label_test, ptest)
        
                wandb.log({
                    "sample/rgb": [wandb.Image(rgb, caption=f"RGB at iter {it}")],
                    "sample/depth": [wandb.Image(depth, caption=f"Depth at iter {it}")],
                    "sample/acc": [wandb.Image(acc, caption=f"Acc at iter {it}")],
                    # "visualization/coordinate_system": wandb.Image(coordinate_viz_path, caption=f"座標系統 {it}"),
                    "epoch_idx": epoch_idx,
                    "iteration": it
                })

             # (v) Compute fid if necessary
            if fid_every > 0 and ((it + 1) % fid_every) == 0:
                fid, kid = evaluator.compute_fid_kid(label)
                wandb.log({
                        "validation/fid": fid,
                        "validation/kid": kid,
                        "iteration": it
                    })
                torch.cuda.empty_cache()
                # save best model
                if save_best == 'fid' and fid < fid_best:
                    fid_best = fid
                    print('Saving best model based on FID...')
                    wandb.run.summary["best_fid"] = fid_best  # 更新 summary
                    checkpoint_io.save('model_best.pt', it=it, epoch_idx=epoch_idx, fid_best=fid_best, kid_best=kid_best, save_to_wandb=True)
                    # logger.save_stats('stats_best.p')
                    torch.cuda.empty_cache()
                
                elif save_best == 'kid' and kid < kid_best:
                    kid_best = kid
                    print('Saving best model based on KID...')
                    wandb.run.summary["best_kid"] = kid_best  # 更新 summary
                    checkpoint_io.save('model_best.pt', it=it, epoch_idx=epoch_idx, fid_best=fid_best, kid_best=kid_best, save_to_wandb=True)
                    # logger.save_stats('stats_best.p')
                    torch.cuda.empty_cache()

            # (i) Backup if necessary
            if ((it + 1) % 10000) == 0:
                print('Saving backup...')
                checkpoint_io.save('model_%08d.pt' % it, it=it, epoch_idx=epoch_idx, save_to_wandb=True)

            # 儲存檢查點
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