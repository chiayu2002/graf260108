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
from contextlib import contextmanager

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
    aux_loss_weight = 0.02

    use_amp = config['training'].get('use_amp', False)
    amp_dtype_str = config['training'].get('amp_dtype', 'bfloat16')
    amp_dtype = torch.bfloat16 if amp_dtype_str == 'bfloat16' else torch.float16

    print(f"[Training] AMP = {use_amp}, dtype = {amp_dtype_str}")

    # GradScaler 只在 fp16 下需要,bf16 不需要(bf16 動態範圍夠)
    if use_amp and amp_dtype == torch.float16:
        scaler = torch.cuda.amp.GradScaler()
        use_scaler = True
    else:
        scaler = None
        use_scaler = False


    # ========================================================
    # [CCSR Toggle] 一個 flag 控制全部 CCSR 相關邏輯
    # ========================================================
    use_ccsr = config['ccsr']['enabled']
    lambda_ccsr = config['ccsr'].get('alpha_init', 1.0) if use_ccsr else 0.0
    print(f"[CCSR] enabled = {use_ccsr}, lambda = {lambda_ccsr}")
    
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

    print("\n[Sanity Check] Hidden state differences:")
    hs_keys = list(cached_hidden_states.keys())
    for i, k1 in enumerate(hs_keys):
        for k2 in hs_keys[i+1:]:
            diff = (cached_hidden_states[k1] - cached_hidden_states[k2]).norm().item()
            cos_sim = torch.nn.functional.cosine_similarity(
                cached_hidden_states[k1].unsqueeze(0),
                cached_hidden_states[k2].unsqueeze(0)
            ).item()
            print(f"  {k1} vs {k2}:  L2={diff:.4f}, cos_sim={cos_sim:.4f}")
    print()

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

    # ========================================================
    # [CCSR Helper] 把 image-format tensor [B,3,H,W] 轉成 D 接受的
    # flat-patch format [B*H*W, 3]
    # ========================================================
    def ccsr_to_flat(x):
        """[B, 3, H, W] -> [B*H*W, 3] 對齊 NeRF rays_to_output 的格式"""
        return x.permute(0, 2, 3, 1).reshape(-1, 3)

    @contextmanager
    def amp_context():
        if use_amp:
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                yield
        else:
            yield

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

            # ==================== Discriminator Step ====================
            d_optimizer.zero_grad()

            rgbs = img_to_patch(x_real)
            rgbs.requires_grad_(True)

            z = zdist.sample((batch_size,))

            # Real — 在 autocast 裡算 d_real 跟 aux loss
            with amp_context():
                d_real, aux_real = discriminator(rgbs, hidden_state, return_aux=True)
                dloss_real = compute_loss(d_real, 1)
                aux_loss_real = F.mse_loss(aux_real, hidden_state)  # 預測 ground truth condition

            # ========================================================
            # [重要] R1 gradient penalty 強制 fp32 — 二階梯度在 fp16/bf16 下
            # 可能 NaN 或數值不穩,即使 bf16 也保守起見走 fp32
            # ========================================================
            reg = reg_param * compute_grad2(d_real.float(), rgbs).mean()

            # Fake
            with torch.no_grad(), amp_context():
                if use_ccsr:
                    rgb_nerf, _, ccsr_out = generator(
                        z, label, hidden_state, return_ccsr_output=True
                    )
                    x_fake_for_d = ccsr_to_flat(ccsr_out)
                else:
                    rgb_nerf, _ = generator(z, label, hidden_state)
                    x_fake_for_d = rgb_nerf

            with amp_context():
                d_fake = discriminator(rgb_nerf, hidden_state)   # 不要 return_aux
                dloss_fake = compute_loss(d_fake, 0)

            total_d_loss = dloss_real + dloss_fake + reg + aux_loss_weight * (aux_loss_real)

            if use_scaler:
                scaler.scale(total_d_loss).backward()
                scaler.step(d_optimizer)
                scaler.update()
            else:
                total_d_loss.backward()
                d_optimizer.step()

            d_scheduler.step()

            # ==================== Generator Step ====================
            if config['nerf']['decrease_noise']:
                generator.decrease_nerf_noise(it)

            toggle_grad(generator, True)
            toggle_grad(discriminator, False)
            generator.train()
            discriminator.train()
            g_optimizer.zero_grad()

            z = zdist.sample((batch_size,))

            with amp_context():
                if use_ccsr:
                    rgb_nerf, _, ccsr_out = generator(
                        z, label, hidden_state, return_ccsr_output=True
                    )
                    ccsr_flat = ccsr_to_flat(ccsr_out)
                    # D 看精修後的版本
                    d_fake = discriminator(ccsr_flat, hidden_state)
                    gloss_adv = compute_loss(d_fake, 1)
                    # Consistency: CCSR 不能跟 NeRF 差太多
                    consistency_loss = F.l1_loss(ccsr_flat, rgb_nerf)
                    gloss = gloss_adv + lambda_ccsr * consistency_loss
                else:
                    rgb_nerf, _ = generator(z, label, hidden_state)
                    # d_fake = discriminator(rgb_nerf, hidden_state)
                    d_fake, aux_fake = discriminator(rgb_nerf, hidden_state, return_aux=True)
                    gloss_adv = compute_loss(d_fake, 1)
                    gloss_aux = F.mse_loss(aux_fake, hidden_state)
                    consistency_loss = torch.tensor(0.0, device=device)  # 佔位
                    gloss = gloss_adv + aux_loss_weight * gloss_aux

            if use_scaler:
                scaler.scale(gloss).backward()
                scaler.step(g_optimizer)
                scaler.update()
            else:
                gloss.backward()
                g_optimizer.step()

            g_scheduler.step()

            current_lr_g = g_optimizer.param_groups[0]['lr']
            current_lr_d = d_optimizer.param_groups[0]['lr']

            if (it + 1) % config['training']['print_every'] == 0:
                log_dict = {
                    "loss/generator_total": gloss.item(),
                    "loss/generator_adv": gloss_adv.item(),
                    "loss/generator_label": gloss_aux.item(),
                    "loss/discriminator": total_d_loss.item(),
                    "loss/discriminator_reallabel": aux_loss_real.item(),
                    # "loss/discriminator_fakelabel": aux_loss_fake.item(),
                    "loss/dloss_real": dloss_real.item(),
                    "loss/dloss_fake": dloss_fake.item(),
                    "loss/regularizer": reg.item(),
                    "learning rate/generator": current_lr_g,
                    "learning rate/discriminator": current_lr_d,
                    "iteration": it,
                }
                if use_ccsr:
                    log_dict["loss/ccsr_consistency"] = consistency_loss
                wandb.log(log_dict)

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
                    "sample/rgb": wandb.Image(grid_rgb, caption=f"iter {it}"),
                    "sample/depth": wandb.Image(grid_depth, caption=f"Depth at iter {it}"),
                    "sample/acc": wandb.Image(grid_acc, caption=f"Acc at iter {it}"),
                    "epoch_idx": epoch_idx,
                    "iteration": it
                })

            # FID/KID
            if fid_every > 0 and ((it + 1) % fid_every) == 0:

                # 收集所有可用的 (label_7d, hidden_state) pairs
                all_conditions = []
                for spec_name, hs in cached_hidden_states.items():
                    spec_key = spec_name + '_n'
                    if spec_key in train_dataset.specimen_props:
                        props = train_dataset.specimen_props[spec_key]
                        label_vec_7d = props['AR'] + props['LR'] + props['TR']
                        all_conditions.append((
                            torch.tensor(label_vec_7d, dtype=torch.float32),
                            hs
                        ))

                # 解析 v_list,知道有幾個 height 可選
                v_list = [float(x.strip()) for x in config['data']['v'].split(",")]
                n_heights = len(v_list)

                def matched_sample_gen():
                    while True:
                        # 隨機抽 condition
                        cond_idxs = np.random.randint(0, len(all_conditions), size=batch_size)
                        label_7d_list = [all_conditions[i][0] for i in cond_idxs]
                        hs_list = [all_conditions[i][1] for i in cond_idxs]

                        # 為每個 sample 隨機抽 view (height_idx, angle_idx)
                        # 補成 9 維 label: [AR(2), LR(3), TR(2), height_idx, angle_idx]
                        full_labels = []
                        for label_7d in label_7d_list:
                            height_idx = float(np.random.randint(0, n_heights))
                            angle_idx = float(np.random.randint(0, 360))
                            view_part = torch.tensor([height_idx, angle_idx], dtype=torch.float32)
                            full_label = torch.cat([label_7d, view_part], dim=0)
                            full_labels.append(full_label)

                        label_batch = torch.stack(full_labels).to(device)   # [B, 9]
                        hs_batch = torch.stack(hs_list).to(device)          # [B, 1024]

                        z = zdist.sample((batch_size,))
                        with torch.no_grad():
                            rgb, _, _ = evaluator.create_samples(z, label_batch, hs_batch)
                        rgb = (rgb / 2 + 0.5).mul_(255).clamp_(0, 255).to(torch.uint8).to(torch.float) / 255. * 2 - 1
                        yield rgb.cpu()

                fid, kid = evaluator.compute_fid_kid(None, None, sample_generator=matched_sample_gen())
                wandb.log({"validation/fid": fid, "validation/kid": kid, "iteration": it})
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