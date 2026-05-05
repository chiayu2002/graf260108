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
from graf.train_step import compute_grad2, compute_loss, toggle_grad, CCSRLoss
from graf.transforms import ImgToPatch

from GAN_stability.gan_training.checkpoints_mod import CheckpointIO


# ========================================================
# Helper functions
# ========================================================

def setup_directories(config):
    out_dir = os.path.join(config['training']['outdir'], config['expname'])
    checkpoint_dir = os.path.join(out_dir, 'chkpts')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    return out_dir, checkpoint_dir


def initialize_training(config, device):
    extractor_path = config['data']['extractor_path']
    extractor_args = json.load(open(glob.glob(
        "/Data/home/vicky/graf260108_im64/HystereticGRU/2026-03-24_17-13-01/args.json",
        recursive=True)[0], "r"))
    extractor_args = argparse.Namespace(**extractor_args)
    extractor = importlib.import_module(
        f"graf.models.HystereticPrediction"
    ).__dict__[extractor_args.architecture](**vars(extractor_args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = extractor.to(device)
    state_dict = torch.load(glob.glob(extractor_path, recursive=True)[0])["state_dict"]
    status = extractor.load_state_dict(state_dict)
    print("Extractor Loading Status: ", status)

    train_dataset, hwfr = get_data(config, extractor, extractor_args)

    if config['data']['orthographic']:
        hw_ortho = (config['data']['far'] - config['data']['near'],) * 2
        hwfr[2] = hw_ortho
    config['data']['hwfr'] = hwfr

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['nworkers'],
        shuffle=True, pin_memory=True,
        sampler=None, drop_last=True,
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
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def extract_real_patches(x_real, all_hw, target_size, device):
    """
    用 FlexGridRaySampler 的 hw 座標從 x_real 提取空間對齊的 patch。
    
    Args:
        x_real:      [B, 3, 256, 256], range [-1, 1]
        all_hw:      list of [patch_h, patch_w, 2] — grid_sample 座標
        target_size: (H, W), e.g. (64, 64)
        device:      torch device
    Returns:
        [B, 3, H, W], range [-1, 1], 與 NeRF 渲染空間完全對齊
    """
    real_patches = []
    for i in range(len(all_hw)):
        hw_i = all_hw[i]  # [patch_h, patch_w, 2]
        # 轉成 [1, 2, ph, pw] → interpolate 到目標尺寸 → [1, tH, tW, 2]
        grid = hw_i.permute(2, 0, 1).unsqueeze(0).to(device)
        grid = F.interpolate(grid, size=target_size, mode='bilinear', align_corners=True)
        grid = grid.squeeze(0).permute(1, 2, 0).unsqueeze(0)
        patch_i = F.grid_sample(
            x_real[i:i+1], grid, mode='bilinear', align_corners=True
        )
        real_patches.append(patch_i)
    return torch.cat(real_patches, dim=0)


# ========================================================
# Main
# ========================================================

def main():
    set_random_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    # ============ 新增：checkpoint resume 參數 ============
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint file to resume training. '
                             'Can be absolute path or filename in checkpoint_dir. '
                             'E.g.: --resume model_best.pt  or  --resume /path/to/model_00010000.pt')
    parser.add_argument('--resume_wandb_id', type=str, default=None,
                        help='wandb run id to resume logging into the same run. '
                             'E.g.: --resume_wandb_id abc123xy')
    # =====================================================
    args = parser.parse_args()

    config = load_config(args.config)
    config['data']['fov'] = float(config['data']['fov'])
    restart_every = config['training']['restart_every']
    batch_size = config['training']['batch_size']
    fid_every = config['training']['fid_every']
    save_best = config['training']['save_best']
    reg_param = config['training']['reg_param']
    device = torch.device("cuda:0")
    aux_loss_weight = config['training']['label_param']

    use_amp = config['training'].get('use_amp', False)
    amp_dtype_str = config['training'].get('amp_dtype', 'bfloat16')
    amp_dtype = torch.bfloat16 if amp_dtype_str == 'bfloat16' else torch.float16
    print(f"[Training] AMP = {use_amp}, dtype = {amp_dtype_str}")

    if use_amp and amp_dtype == torch.float16:
        scaler = torch.cuda.amp.GradScaler()
        use_scaler = True
    else:
        scaler = None
        use_scaler = False

    # ========================================================
    # Reconstruction Loss 設定
    # ========================================================
    recon_config = config.get('reconstruction', {})
    use_recon = recon_config.get('enabled', False)
    lambda_recon = recon_config.get('lambda_recon', 0.1)
    print(f"[Recon] enabled={use_recon}, lambda_recon={lambda_recon}")
    if use_recon:
        print(f"[Recon] G_loss = GAN_adv + {lambda_recon} × Recon(NeRF, real) + aux")

    out_dir, checkpoint_dir = setup_directories(config)
    save_config(os.path.join(out_dir, 'config.yaml'), config)
    
    # ============ 修改：wandb init 支援 resume ============
    wandb_kwargs = dict(
        project=config['wandb']['project'],
        name=config['wandb']['name'],
        config=config,
    )
    if args.resume_wandb_id is not None:
        wandb_kwargs['id'] = args.resume_wandb_id
        wandb_kwargs['resume'] = 'must'
        print(f"[wandb] Resuming run id={args.resume_wandb_id}")
    wandb.init(**wandb_kwargs)
    # =====================================================

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

    # Reconstruction loss: 跟 CCSRLoss 同一個東西（L1 + VGG perceptual）
    # 但這次直接用在 NeRF 輸出上
    if use_recon:
        recon_loss_fn = CCSRLoss(device=device)
        print(f"[Recon] Loss = L1 + 0.5 × VGG perceptual")

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
        evaluator.inception_eval.initialize_target(
            val_loader, cache_file=fid_cache_file, act_cache_file=kid_cache_file
        )
    
    cached_hidden_states = {}
    for exp_name, hs in train_dataset.hidden_state.items():
        cached_hidden_states[exp_name] = hs.to(device)
    print(f"Cached hidden states for: {list(cached_hidden_states.keys())}")

    print("\n[Sanity Check] Hidden state differences:")
    hs_keys = list(cached_hidden_states.keys())
    for i, k1 in enumerate(hs_keys):
        for k2 in hs_keys[i+1:]:
            diff = (cached_hidden_states[k1] - cached_hidden_states[k2]).norm().item()
            cos_sim = F.cosine_similarity(
                cached_hidden_states[k1].unsqueeze(0),
                cached_hidden_states[k2].unsqueeze(0)
            ).item()
            print(f"  {k1} vs {k2}:  L2={diff:.4f}, cos_sim={cos_sim:.4f}")
    print()

    fid_best = float('inf')
    kid_best = float('inf')
    it = epoch_idx = -1
    tstart = t0 = time.time()

    # ========================================================
    # 新增：載入 checkpoint 恢復訓練
    # ========================================================
    if args.resume is not None:
        print(f"\n[Resume] Loading checkpoint: {args.resume}")
        try:
            load_scalars = checkpoint_io.load(args.resume)
            # 恢復 iteration 和 epoch
            it = load_scalars.get('it', -1)
            epoch_idx = load_scalars.get('epoch_idx', -1)
            fid_best = load_scalars.get('fid_best', float('inf'))
            kid_best = load_scalars.get('kid_best', float('inf'))
            print(f"[Resume] Restored: it={it}, epoch_idx={epoch_idx}, "
                  f"fid_best={fid_best:.4f}, kid_best={kid_best:.4f}")
            print(f"[Resume] Training will continue from iteration {it + 1}")
        except FileNotFoundError:
            print(f"[Resume] ERROR: Checkpoint file '{args.resume}' not found!")
            print(f"[Resume] Looked in: {checkpoint_dir}")
            print(f"[Resume] Available checkpoints:")
            ckpt_files = glob.glob(os.path.join(checkpoint_dir, '*.pt'))
            for f in sorted(ckpt_files):
                print(f"  - {os.path.basename(f)}")
            if not ckpt_files:
                print("  (none found)")
            raise
    # ========================================================
    
    g_scheduler = build_lr_scheduler(g_optimizer, config, last_epoch=it)
    d_scheduler = build_lr_scheduler(d_optimizer, config, last_epoch=it)

    @contextmanager
    def amp_context():
        if use_amp:
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                yield
        else:
            yield

    v_list = [float(x.strip()) for x in config['data']['v'].split(",")]
    n_heights = len(v_list)

    # ========================================================
    # Training Loop
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

            # ==================== Discriminator Step ====================
            # （跟原本完全一樣，不受 reconstruction loss 影響）
            d_optimizer.zero_grad()

            rgbs = img_to_patch(x_real)
            rgbs.requires_grad_(True)

            z = zdist.sample((batch_size,))

            with amp_context():
                d_real, aux_real = discriminator(rgbs, hidden_state, return_aux=True)
                dloss_real = compute_loss(d_real, 1)
                aux_loss_real = F.mse_loss(aux_real, hidden_state)

            reg = reg_param * compute_grad2(d_real.float(), rgbs).mean()

            with torch.no_grad(), amp_context():
                rgb_nerf, _ = generator(z, label, hidden_state)
                x_fake_for_d = rgb_nerf

            with amp_context():
                d_fake = discriminator(x_fake_for_d, hidden_state)
                dloss_fake = compute_loss(d_fake, 0)

            total_d_loss = dloss_real + dloss_fake + reg + aux_loss_weight * aux_loss_real

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
                if use_recon:
                    # ====================================================
                    # GAN + Reconstruction Loss
                    #
                    # 手動 sample rays → 記住 hw 座標
                    # → NeRF 渲染 64×64
                    # → 用 hw 從 x_real 提取同位置的 64×64 patch
                    # → L1 + VGG loss
                    # ====================================================

                    # Step 1: 手動 sample rays，記住 hw
                    all_rays = []
                    all_hw = []
                    for i in range(batch_size):
                        height_idx = int(label[i, 7].item())
                        angle_idx = int(label[i, 8].item())
                        selected_u = angle_idx / 360
                        selected_v = v_list[height_idx % n_heights]
                        pose_i = generator.sample_select_pose(selected_u, selected_v)
                        batch_rays_i, _, hw_i = generator.ray_sampler(
                            generator.H, generator.W, generator.focal, pose_i
                        )
                        all_rays.append(batch_rays_i)
                        all_hw.append(hw_i)
                    rays = torch.cat(all_rays, dim=1)

                    # Step 2: NeRF 渲染（用外部 rays，帶梯度）
                    rgb_nerf, _ = generator(z, label, hidden_state, rays=rays)
                    # rgb_nerf: [B*4096, 3], range [-1, 1]

                    # Step 3: GAN loss
                    d_fake, aux_fake = discriminator(
                        rgb_nerf, hidden_state, return_aux=True
                    )
                    gloss_adv = compute_loss(d_fake, 1)
                    gloss_aux = F.mse_loss(aux_fake, hidden_state)

                    # Step 4: Reconstruction loss
                    # reshape flat → 2D patch（保留梯度到 NeRF）
                    patch_size = int(np.sqrt(rgb_nerf.shape[0] // batch_size))
                    nerf_patch = rgb_nerf.view(
                        batch_size, patch_size, patch_size, 3
                    ).permute(0, 3, 1, 2).contiguous()
                    # nerf_patch: [B, 3, 64, 64], range [-1, 1]

                    # 從 x_real 提取同位置的 real patch
                    real_patch = extract_real_patches(
                        x_real, all_hw,
                        target_size=(patch_size, patch_size),
                        device=device
                    )
                    # real_patch: [B, 3, 64, 64], range [-1, 1]

                    recon_loss = recon_loss_fn(nerf_patch, real_patch)

                    # Step 5: 加總
                    gloss = (gloss_adv
                             + lambda_recon * recon_loss
                             + aux_loss_weight * gloss_aux)

                else:
                    # ====================================================
                    # 原始 GAN-only（無 reconstruction loss）
                    # ====================================================
                    rgb_nerf, _ = generator(z, label, hidden_state)
                    d_fake, aux_fake = discriminator(
                        rgb_nerf, hidden_state, return_aux=True
                    )
                    gloss_adv = compute_loss(d_fake, 1)
                    gloss_aux = F.mse_loss(aux_fake, hidden_state)
                    recon_loss = torch.tensor(0.0, device=device)
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

            # ==================== Logging ====================
            # 修正：用一個 dict 收集所有 log，最後只呼叫一次 wandb.log()
            # 避免同一個 iteration 多次 log 造成重複 panel
            log_dict = None

            if (it + 1) % config['training']['print_every'] == 0:
                log_dict = {
                    "loss/generator_total": gloss.item(),
                    "loss/generator_adv": gloss_adv.item(),
                    "loss/generator_label": gloss_aux.item(),
                    "loss/discriminator": total_d_loss.item(),
                    "loss/discriminator_reallabel": aux_loss_real.item(),
                    "loss/dloss_real": dloss_real.item(),
                    "loss/dloss_fake": dloss_fake.item(),
                    "loss/regularizer": reg.item(),
                    "learning rate/generator": current_lr_g,
                    "learning rate/discriminator": current_lr_d,
                }
                if use_recon:
                    log_dict["loss/recon"] = recon_loss.item()

            # ==================== Sample ====================
            if ((it % config['training']['sample_every']) == 0) or \
               ((it < 5000) and (it % 200 == 0)):
                angle_positions = [(i / 8, 0.5) for i in range(8)]
                plist = [generator.sample_select_pose(u, v)
                         for (u, v) in angle_positions]
                ptest = torch.stack(plist)
                angles = [0, 45, 90, 135, 180, 225, 270, 315]

                specimens_to_sample = {
                    'RS307': [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    'RS330': [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                    'RS615': [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                    'RS315': [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                }

                ztest = zdist.sample((8,))
                rgb_panels, depth_panels, acc_panels = [], [], []
                spec_names_shown = []

                for spec_name, vec in specimens_to_sample.items():
                    if spec_name not in cached_hidden_states:
                        continue
                    test_labels_list = [vec + [0.5, float(a)] for a in angles]
                    label_test = torch.tensor(
                        test_labels_list, dtype=torch.float32
                    ).to(device)
                    hs_spec = cached_hidden_states[spec_name]
                    hs_test = hs_spec.unsqueeze(0).expand(8, -1)

                    rgb, depth, acc = evaluator.create_samples(
                        ztest.to(device), label_test, hs_test, ptest
                    )

                    if depth.dim() == 3:
                        depth = depth.unsqueeze(1)
                    if depth.shape[1] == 1:
                        depth = depth.expand(-1, 3, -1, -1)
                    if acc.dim() == 3:
                        acc = acc.unsqueeze(1)
                    if acc.shape[1] == 1:
                        acc = acc.expand(-1, 3, -1, -1)

                    rgb_panels.append(rgb.detach().cpu())
                    depth_panels.append(depth.detach().cpu())
                    acc_panels.append(acc.detach().cpu())
                    spec_names_shown.append(spec_name)

                if rgb_panels:
                    rgb_all = torch.cat(rgb_panels, dim=0)
                    depth_all = torch.cat(depth_panels, dim=0)
                    acc_all = torch.cat(acc_panels, dim=0)
                    grid_rgb = vutils.make_grid(rgb_all, nrow=8, normalize=True)
                    grid_depth = vutils.make_grid(depth_all, nrow=8, normalize=True)
                    grid_acc = vutils.make_grid(acc_all, nrow=8, normalize=True)
                    caption = f"iter {it} | {' / '.join(spec_names_shown)}"

                    # 合併到同一個 log_dict，不要單獨 wandb.log()
                    if log_dict is None:
                        log_dict = {}
                    log_dict["sample/rgb"] = wandb.Image(grid_rgb, caption=caption)
                    log_dict["sample/depth"] = wandb.Image(grid_depth, caption=caption)
                    log_dict["sample/acc"] = wandb.Image(grid_acc, caption=caption)

            # ==================== FID/KID ====================
            if fid_every > 0 and ((it + 1) % fid_every == 0):
                def pose_from_label(gen, label_i, v_list_local):
                    height_idx = int(label_i[7].item())
                    angle_idx = int(label_i[8].item())
                    u = angle_idx / 360.0
                    v = v_list_local[height_idx % len(v_list_local)]
                    return gen.sample_select_pose(u, v)

                n_dataset = len(train_dataset)

                def matched_sample_gen():
                    while True:
                        indices = np.random.choice(
                            n_dataset, size=batch_size, replace=False
                        )
                        labels_list, hs_list = [], []
                        for idx in indices:
                            _, label_i, hs_i = train_dataset[idx]
                            labels_list.append(label_i)
                            hs_list.append(hs_i)
                        label_batch = torch.stack(labels_list).to(device)
                        hs_batch = torch.stack(hs_list).to(device)
                        poses = torch.stack([
                            pose_from_label(generator, label_batch[i], v_list)
                            for i in range(batch_size)
                        ])
                        z_fid = zdist.sample((batch_size,))
                        with torch.no_grad():
                            rgb, _, _ = evaluator.create_samples(
                                z_fid, label_batch, hs_batch, poses
                            )
                        rgb = (rgb / 2 + 0.5).mul_(255).clamp_(0, 255) \
                              .to(torch.uint8).to(torch.float) / 255. * 2 - 1
                        yield rgb.cpu()

                fid, kid = evaluator.compute_fid_kid(
                    None, None, sample_generator=matched_sample_gen()
                )

                # 合併到同一個 log_dict
                if log_dict is None:
                    log_dict = {}
                log_dict["validation/fid"] = fid
                log_dict["validation/kid"] = kid

                torch.cuda.empty_cache()

                if save_best == 'fid' and fid < fid_best:
                    fid_best = fid
                    print('Saving best model based on FID...')
                    wandb.run.summary["best_fid"] = fid_best
                    checkpoint_io.save(
                        'model_best.pt', it=it, epoch_idx=epoch_idx,
                        fid_best=fid_best, kid_best=kid_best,
                        save_to_wandb=True
                    )
                    torch.cuda.empty_cache()
                elif save_best == 'kid' and kid < kid_best:
                    kid_best = kid
                    print('Saving best model based on KID...')
                    wandb.run.summary["best_kid"] = kid_best
                    checkpoint_io.save(
                        'model_best.pt', it=it, epoch_idx=epoch_idx,
                        fid_best=fid_best, kid_best=kid_best,
                        save_to_wandb=True
                    )
                    torch.cuda.empty_cache()

            # ==================== 統一 wandb.log() ====================
            # 每個 iteration 最多只呼叫一次 wandb.log()，用 step=it 固定 x 軸
            if log_dict is not None:
                wandb.log(log_dict, step=it)

            # ==================== Checkpoint ====================
            if ((it + 1) % 10000) == 0:
                print('Saving backup...')
                checkpoint_io.save(
                    'model_%08d.pt' % it, it=it, epoch_idx=epoch_idx,
                    fid_best=fid_best, kid_best=kid_best,
                    save_to_wandb=True
                )

            if time.time() - t0 > config['training']['save_every']:
                checkpoint_io.save(
                    config['training']['model_file'],
                    it=it, epoch_idx=epoch_idx,
                    fid_best=fid_best, kid_best=kid_best,
                    save_to_wandb=True
                )
                t0 = time.time()
                if (restart_every > 0 and t0 - tstart > restart_every):
                    return


if __name__ == '__main__':
    main()