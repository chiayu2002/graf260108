import argparse
import os
from os import path
import numpy as np
import time
import copy
import csv
import random
import json
import glob
import importlib
import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.append('submodules')

from submodules.GAN_stability.gan_training.checkpoints_mod import CheckpointIO
from graf.gan_training import Evaluator
from graf.config import get_data, build_models, get_render_poses, load_config
from graf.utils import count_trainable_parameters, to_phi, to_theta, get_zdist
from graf.transforms import ImgToPatch


# ============================================================
# 指標計算函數
# ============================================================

def compute_psnr(img_pred, img_gt):
    """
    計算 Peak Signal-to-Noise Ratio (PSNR)
    
    Args:
        img_pred: 預測影像 [C, H, W] 或 [H, W, C]，範圍 [0, 1]
        img_gt:   真實影像 [C, H, W] 或 [H, W, C]，範圍 [0, 1]
    
    Returns:
        float: PSNR 值 (dB)
    """
    mse = torch.mean((img_pred - img_gt) ** 2).item()
    if mse < 1e-10:
        return float('inf')
    psnr = 10 * np.log10(1.0 / mse)  # MAX=1 for [0,1] range
    return psnr


def compute_r_squared(img_pred, img_gt):
    """
    計算 R² (Coefficient of Determination)
    
    R² = 1 - SS_res / SS_tot
    SS_res = Σ(y_true - y_pred)²
    SS_tot = Σ(y_true - y_mean)²
    
    R² = 1.0 表示完美預測
    R² = 0.0 表示和平均值一樣好
    R² < 0   表示比平均值還差
    
    Args:
        img_pred: 預測影像，展平後計算 [N]
        img_gt:   真實影像，展平後計算 [N]
    
    Returns:
        float: R² 值
    """
    pred_flat = img_pred.flatten().float()
    gt_flat = img_gt.flatten().float()
    
    ss_res = torch.sum((gt_flat - pred_flat) ** 2).item()
    ss_tot = torch.sum((gt_flat - torch.mean(gt_flat)) ** 2).item()
    
    if ss_tot < 1e-10:
        return float('nan')
    
    r2 = 1.0 - ss_res / ss_tot
    return r2


def compute_ssim(img_pred, img_gt, window_size=11, C1=0.01**2, C2=0.03**2):
    """
    計算 Structural Similarity Index (SSIM) — 簡易版
    
    Args:
        img_pred: [C, H, W]，範圍 [0, 1]
        img_gt:   [C, H, W]，範圍 [0, 1]
    
    Returns:
        float: SSIM 值
    """
    import torch.nn.functional as F
    
    # 確保是 4D [B, C, H, W]
    if img_pred.dim() == 3:
        img_pred = img_pred.unsqueeze(0)
        img_gt = img_gt.unsqueeze(0)
    
    C = img_pred.shape[1]
    
    # 建立高斯視窗
    def gaussian_window(size, sigma=1.5):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g.unsqueeze(1) @ g.unsqueeze(0)
    
    window = gaussian_window(window_size).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    window = window.expand(C, -1, -1, -1).to(img_pred.device)
    
    pad = window_size // 2
    
    mu1 = F.conv2d(img_pred, window, padding=pad, groups=C)
    mu2 = F.conv2d(img_gt, window, padding=pad, groups=C)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img_pred * img_pred, window, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img_gt * img_gt, window, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(img_pred * img_gt, window, padding=pad, groups=C) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()


# ============================================================
# 配對生成：給定真實影像的 label，生成對應的假影像
# ============================================================

def generate_paired_samples(generator, evaluator, dataset, zdist, cached_hidden_states, 
                            device, num_pairs=100, seed=42):
    """
    從 dataset 中取出真實影像，並用 generator 在相同的視角和條件下生成假影像。
    
    返回:
        list of dict: 每個 dict 包含 'real', 'fake', 'label', 'exp_name'
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    generator.eval()
    
    v_list = generator._v_list
    pairs = []
    
    # 從 dataset 中均勻取樣
    indices = np.random.choice(len(dataset), size=min(num_pairs, len(dataset)), replace=False)
    
    for idx in indices:
        img_real, label, hidden_state_cpu = dataset[idx]
        
        # img_real: [3, H, W] 範圍 [-1, 1]
        # label: [9] = [AR(2), LR(3), TR(2), height_idx, angle_idx]
        
        label = label.unsqueeze(0).to(device)           # [1, 9]
        hidden_state = hidden_state_cpu.unsqueeze(0).to(device)  # [1, 1024]
        
        # 取得視角參數
        height_idx = int(label[0, 7].item())
        angle_idx = int(label[0, 8].item())
        selected_u = angle_idx / 360
        selected_v = v_list[height_idx]
        
        # 生成假影像
        z = zdist.sample((1,))

        with torch.no_grad():
            pose = generator.sample_select_pose(selected_u, selected_v)
            rays = generator.sample_select_rays(selected_u, selected_v)

            # eval 模式下 generator 回傳 4 個值 (use_test_kwargs=True)
            # rgb_fake shape: [N_rays, 3],範圍已經是 [-1, 1]
            rgb_fake, _, _, _ = generator(z, label, hidden_state, rays=rays)

        # ========================================================
        # [修正] rgb_fake 是 [N_rays, 3],不是 [B, N*3]
        # patch_size = sqrt(N_rays)
        # ========================================================
        n_rays = rgb_fake.shape[0]
        patch_size = int(np.sqrt(n_rays))
        assert patch_size * patch_size == n_rays, \
            f"n_rays={n_rays} is not a perfect square"

        # Denormalize [-1, 1] → [0, 1]
        rgb_fake_01 = (rgb_fake / 2 + 0.5).clamp(0, 1)   # [N_rays, 3]

        # [N_rays, 3] → [H, W, 3] → [3, H, W]
        rgb_fake_img = rgb_fake_01.view(patch_size, patch_size, 3).permute(2, 0, 1)  # [3, H, W]

        # 真實影像 denormalize
        img_real_01 = (img_real / 2 + 0.5).clamp(0, 1)  # [3, H, W]

        # 如果尺寸不同,把真實影像 resize 到 generator 輸出的尺寸
        if img_real_01.shape[1] != patch_size:
            img_real_resized = torch.nn.functional.interpolate(
                img_real_01.unsqueeze(0),
                size=(patch_size, patch_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        else:
            img_real_resized = img_real_01

        # 判斷實驗名稱
        filename = dataset.filenames[idx]
        exp_name = 'unknown'
        for e in dataset.exp_list:
            if e in filename:
                exp_name = e
                break

        pairs.append({
            'real': img_real_resized.cpu(),   # [3, H, W] range [0,1]
            'fake': rgb_fake_img.cpu(),       # [3, H, W] range [0,1]  ← 不用再 squeeze
            'label': label.cpu().squeeze(0),
            'exp_name': exp_name,
        })
    
    generator.train()
    return pairs


# ============================================================
# 主程式
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate GRAF model.')
    parser.add_argument('--config', type=str, default='./results/column20260407_gru_hidden_state_disc_Dbn_lrg5_aux/config.yaml', help='Path to config file.')
    parser.add_argument('--checkpoint', type=str, default='./results/column20260407_gru_hidden_state_disc_Dbn_lrg5_aux/chkpts/model_00039999.pt', help='Path to checkpoint file.')
    parser.add_argument('--fid_kid', action='store_true', help='Evaluate FID and KID.')
    parser.add_argument('--psnr_r2', default=False, action='store_true', help='Evaluate PSNR and R².')
    parser.add_argument('--create_sample', default=True, action='store_true', help='Generate sample images.')
    parser.add_argument('--num_pairs', type=int, default=200, help='Number of pairs for PSNR/R² evaluation.')
    parser.add_argument('--all', default=False , action='store_true', help='Run all evaluations.')
    
    args = parser.parse_args()
    
    if args.all:
        args.fid_kid = True
        args.psnr_r2 = True
        args.create_sample = True
    
    config = load_config(args.config)
    config['data']['fov'] = float(config['data']['fov'])
    
    batch_size = config['training']['batch_size']
    out_dir = os.path.join(config['training']['outdir'], config['expname'])
    checkpoint_dir = path.join(out_dir, 'chkpts')
    eval_dir = os.path.join(out_dir, 'eval')
    os.makedirs(eval_dir, exist_ok=True)
    
    config['training']['nworkers'] = 0
    
    def set_random_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    set_random_seed(0)
    
    device = torch.device("cuda:0")
    
    # ========================================================
    # [修正] 載入 GRU extractor — 原本 eval.py 缺少這段
    # 原本直接呼叫 get_data(config) 不帶 extractor
    # 但 Dataset 需要 extractor 來計算 hidden_state
    # ========================================================
    print("Loading GRU extractor...")
    extractor_path = config['data']['extractor_path']
    extractor_args_path = os.path.dirname(os.path.dirname(os.path.dirname(extractor_path)))
    extractor_args = json.load(open("HystereticGRU/2026-03-24_17-13-01/args.json", "r"))
    extractor_args = argparse.Namespace(**extractor_args)
    extractor = importlib.import_module("graf.models.HystereticPrediction").__dict__[extractor_args.architecture](**vars(extractor_args))
    extractor = extractor.to(device)
    state_dict = torch.load(glob.glob(extractor_path, recursive=True)[0])["state_dict"]
    status = extractor.load_state_dict(state_dict)
    print(f"Extractor Loading Status: {status}")
    
    # ========================================================
    # [修正] 帶上 extractor 和 extractor_args
    # ========================================================
    train_dataset, hwfr = get_data(config, extractor, extractor_args)
    config['data']['hwfr'] = hwfr
    print(f"Dataset loaded: {len(train_dataset)} images, hwfr={hwfr}")
    
    # 預先快取各實驗的 hidden_state
    cached_hidden_states = {}
    for exp_name, hs in train_dataset.hidden_state.items():
        cached_hidden_states[exp_name] = hs.to(device)
    print(f"Cached hidden states for: {list(cached_hidden_states.keys())}")
    
    val_dataset = train_dataset
    
    # ========================================================
    # 建立模型
    # ========================================================
    generator, _ = build_models(config, disc=False)
    print(f'Generator params: {count_trainable_parameters(generator)}')
    generator = generator.to(device)

    img_to_patch = ImgToPatch(generator.ray_sampler, hwfr[:3])

    zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'], device=device)

    evaluator = Evaluator(args.fid_kid, generator, zdist, None,
                        batch_size=batch_size, device=device)

    # ========================================================
    # [修正] CheckpointIO 只建一次,register 後立刻 load
    # 之前的版本建了兩次,第二次把 register 清掉了 → 載入空權重 → 黑圖
    # ========================================================
    checkpoint_io = CheckpointIO(checkpoint_dir=checkpoint_dir)
    checkpoint_io.register_modules(**generator.module_dict)

    # Sanity check: 確認 register 有 module
    print(f"Registered modules in CheckpointIO: {list(checkpoint_io.module_dict.keys())}")
    assert len(checkpoint_io.module_dict) > 0, "No modules registered, checkpoint will load nothing!"

    # 載入 checkpoint — 用 args.checkpoint 而不是 hardcode
    checkpoint_filename = os.path.basename(args.checkpoint)
    print(f"Loading checkpoint: {checkpoint_filename}")
    load_dict = checkpoint_io.load(checkpoint_filename)
    it = load_dict.get('it', -1)
    epoch_idx = load_dict.get('epoch_idx', -1)
    print(f"Loaded checkpoint at iteration {it}, epoch {epoch_idx}")

    # Sanity check: 確認權重真的進去了
    nerf_model = generator.render_kwargs_train['network_fn']
    first_layer_norm = nerf_model.condition_feature.weight.norm().item()
    print(f"[Sanity] condition_feature weight norm = {first_layer_norm:.4f}")
    print("  (如果是 ~10-30 → 載入成功;如果是 ~1-5 → 還是隨機初始化,有問題)")
    
    # ============================================================
    # 1. 生成視覺化樣本
    # ============================================================
    if args.create_sample:
        print("\n" + "="*60)
        print("Generating visual samples...")
        print("="*60)
        
        N_samples = 8
        angle_positions = [(i/8, 0.5) for i in range(8)]
        
        # 定義各實驗的 label
        specimens = {
            'RS307': [1.0, 0.0,  1.0, 0.0, 0.0,  0.0, 1.0],
            'RS330': [1.0, 0.0,  0.0, 0.0, 1.0,  1.0, 0.0],
            'RS615': [0.0, 1.0,  0.0, 1.0, 0.0,  1.0, 0.0],
        }
        
        for spec_name, vec in specimens.items():
            z = zdist.sample((N_samples,))
            
            # [修正] 使用對應實驗的 hidden_state
            if spec_name not in cached_hidden_states:
                print(f"  Skipping {spec_name}: no cached hidden state")
                continue
            hs = cached_hidden_states[spec_name].unsqueeze(0).expand(N_samples, -1)  # [N, 1024]
            
            all_rgb = []
            for i, (u, v) in enumerate(angle_positions):
                angle = int(u * 360)
                label_vec = vec + [0.5, float(angle)]
                label_tensor = torch.tensor([label_vec], dtype=torch.float32).to(device)
                hs_single = hs[i:i+1]
                
                pose = generator.sample_select_pose(u, v)
                
                # [修正] 傳入 hidden_state
                rgb, depth, acc = evaluator.create_samples(
                    z[i:i+1].to(device), label_tensor, hs_single, pose.unsqueeze(0)
                )
                all_rgb.append(rgb)
            
            rgb = torch.cat(all_rgb, dim=0)
            rgb = ((rgb / 2 + 0.5).clamp(0, 1) * 255).to(torch.uint8).float() / 255
            
            filename = f'samples_{spec_name}_iter{it}.png'
            outpath = os.path.join(eval_dir, filename)
            save_image(rgb, outpath, nrow=8)
            print(f"  Saved {spec_name} samples to {outpath}")
    
    # ============================================================
    # 2. FID / KID 計算
    # ============================================================
    if args.fid_kid:
        print("\n" + "="*60)
        print("Computing FID and KID...")
        print("="*60)

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=0,
            shuffle=True,
            pin_memory=False,
            drop_last=False,
            generator=torch.Generator(device='cuda:0')
        )

        # Step 1: 計算真實圖的 Inception 特徵統計量
        fid_cache_file = os.path.join(out_dir, 'fid_cache_train.npz')
        kid_cache_file = os.path.join(out_dir, 'kid_cache_train.npz')
        evaluator.inception_eval.initialize_target(
            val_loader, cache_file=fid_cache_file, act_cache_file=kid_cache_file
        )

        # ========================================================
        # [主要指標] Marginal FID with matched conditioning
        # 對每個 batch 隨機從 dataset 抽 real images，用它們的 hidden_state
        # 來生 fake → fake 的條件分布自然匹配 real
        # ========================================================
        print("\n  [Marginal FID with matched conditioning]")

        def matched_sample_generator():
            """每次 yield 一個 batch 的 fake，條件從 dataset 隨機抽"""
            n_dataset = len(train_dataset)
            while True:
                # 隨機抽 batch_size 個 real samples 拿條件
                indices = np.random.choice(n_dataset, size=batch_size, replace=False)
                labels_batch = []
                hs_batch = []
                for idx in indices:
                    _, label_i, hs_i = train_dataset[idx]
                    labels_batch.append(label_i)
                    hs_batch.append(hs_i)
                label_batch = torch.stack(labels_batch).to(device)       # [B, 9]
                hs_batch = torch.stack(hs_batch).to(device)              # [B, 1024]

                z = zdist.sample((batch_size,))
                with torch.no_grad():
                    rgb, _, _ = evaluator.create_samples(z, label_batch, hs_batch)

                # uint8 quantization (跟 train.py 裡 sample 一致)
                rgb = (rgb / 2 + 0.5).mul_(255).clamp_(0, 255).to(torch.uint8).to(torch.float) / 255. * 2 - 1
                yield rgb.cpu()

        fid_marginal, kid_marginal = evaluator.compute_fid_kid(
            real_label=None,  # 不會被用到
            hidden_state=None,  # 不會被用到
            sample_generator=matched_sample_generator()
        )
        print(f"    Marginal FID = {fid_marginal:.2f}")
        print(f"    Marginal KID×100 = {kid_marginal*100:.2f}")
        torch.cuda.empty_cache()

        # ========================================================
        # [Diagnostic] Per-specimen FID — 看每個材料條件單獨表現
        # 注意：每個 spec 的樣本數少，數字會有 small-sample bias，
        # 只當作相對比較用，不要當絕對指標
        # ========================================================
        print("\n  [Per-specimen FID (diagnostic only — biased high due to small n)]")
        fid_results = {}
        kid_results = {}

        # 解析 v_list
        v_list_eval = [float(x.strip()) for x in config['data']['v'].split(",")]
        n_heights_eval = len(v_list_eval)

        for spec_name, hs in cached_hidden_states.items():
            spec_key = spec_name + '_n'
            if spec_key not in train_dataset.specimen_props:
                continue

            props = train_dataset.specimen_props[spec_key]
            label_vec_7d = props['AR'] + props['LR'] + props['TR']

            # ========================================================
            # 補成 9 維 — 因為 per-spec FID 是要評估這個 spec 在
            # 「各種視角」下的整體品質,所以每次 sample 隨機抽 view
            # ========================================================
            def per_spec_sample_gen(label_7d, hs_single):
                while True:
                    full_labels = []
                    for _ in range(batch_size):
                        h_idx = float(np.random.randint(0, n_heights_eval))
                        a_idx = float(np.random.randint(0, 360))
                        full_label = label_7d + [h_idx, a_idx]
                        full_labels.append(torch.tensor(full_label, dtype=torch.float32))
                    label_batch = torch.stack(full_labels).to(device)
                    hs_batch = hs_single.unsqueeze(0).expand(batch_size, -1).to(device)

                    z = zdist.sample((batch_size,))
                    with torch.no_grad():
                        rgb, _, _ = evaluator.create_samples(z, label_batch, hs_batch)
                    rgb = (rgb / 2 + 0.5).mul_(255).clamp_(0, 255).to(torch.uint8).to(torch.float) / 255. * 2 - 1
                    yield rgb.cpu()

            print(f"    Computing for {spec_name}...")
            fid, kid = evaluator.compute_fid_kid(
                None, None,
                sample_generator=per_spec_sample_gen(label_vec_7d, hs)
            )
            fid_results[spec_name] = fid
            kid_results[spec_name] = kid
            print(f"      FID={fid:.2f}, KID×100={kid*100:.2f}")
            torch.cuda.empty_cache()

        # 儲存結果
        filename = f'fid_kid_iter{it}.csv'
        outpath = os.path.join(eval_dir, filename)
        with open(outpath, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['metric', 'fid', 'kid', 'kid_x100'])
            writer.writerow(['MARGINAL (primary)',
                            f"{fid_marginal:.4f}",
                            f"{kid_marginal:.6f}",
                            f"{kid_marginal*100:.4f}"])
            writer.writerow([])
            writer.writerow(['per-spec (diagnostic)', 'fid', 'kid', 'kid_x100'])
            for spec_name in fid_results:
                writer.writerow([spec_name,
                                f"{fid_results[spec_name]:.4f}",
                                f"{kid_results[spec_name]:.6f}",
                                f"{kid_results[spec_name]*100:.4f}"])
        print(f"\n  Saved to {outpath}")
    
    # ============================================================
    # 3. PSNR / R² / SSIM 計算
    # ============================================================
    if args.psnr_r2:
        print("\n" + "="*60)
        print(f"Computing PSNR, R², SSIM ({args.num_pairs} pairs)...")
        print("="*60)
        
        # 生成配對樣本
        pairs = generate_paired_samples(
            generator, evaluator, train_dataset, zdist,
            cached_hidden_states, device,
            num_pairs=args.num_pairs, seed=42
        )
        
        # 按實驗分組計算指標
        metrics_by_exp = {}
        all_psnr = []
        all_r2 = []
        all_ssim = []
        
        for pair in pairs:
            real = pair['real']   # [3, H, W] range [0,1]
            fake = pair['fake']   # [3, H, W] range [0,1]
            exp_name = pair['exp_name']
            
            psnr_val = compute_psnr(fake, real)
            r2_val = compute_r_squared(fake, real)
            ssim_val = compute_ssim(fake, real)
            
            if exp_name not in metrics_by_exp:
                metrics_by_exp[exp_name] = {'psnr': [], 'r2': [], 'ssim': []}
            
            metrics_by_exp[exp_name]['psnr'].append(psnr_val)
            metrics_by_exp[exp_name]['r2'].append(r2_val)
            metrics_by_exp[exp_name]['ssim'].append(ssim_val)
            
            all_psnr.append(psnr_val)
            all_r2.append(r2_val)
            all_ssim.append(ssim_val)
        
        # 印出結果
        print(f"\n{'Experiment':<12} {'PSNR (dB)':<12} {'R²':<12} {'SSIM':<12} {'Count':<8}")
        print("-" * 56)
        
        for exp_name in sorted(metrics_by_exp.keys()):
            m = metrics_by_exp[exp_name]
            print(f"{exp_name:<12} "
                  f"{np.mean(m['psnr']):>8.2f}±{np.std(m['psnr']):.2f}  "
                  f"{np.mean(m['r2']):>8.4f}±{np.std(m['r2']):.4f}  "
                  f"{np.mean(m['ssim']):>8.4f}±{np.std(m['ssim']):.4f}  "
                  f"{len(m['psnr']):>4}")
        
        print("-" * 56)
        print(f"{'OVERALL':<12} "
              f"{np.mean(all_psnr):>8.2f}±{np.std(all_psnr):.2f}  "
              f"{np.mean(all_r2):>8.4f}±{np.std(all_r2):.4f}  "
              f"{np.mean(all_ssim):>8.4f}±{np.std(all_ssim):.4f}  "
              f"{len(all_psnr):>4}")
        
        # 儲存 CSV
        filename = f'psnr_r2_ssim_iter{it}.csv'
        outpath = os.path.join(eval_dir, filename)
        with open(outpath, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['experiment', 'psnr_mean', 'psnr_std', 'r2_mean', 'r2_std', 
                           'ssim_mean', 'ssim_std', 'count'])
            for exp_name in sorted(metrics_by_exp.keys()):
                m = metrics_by_exp[exp_name]
                writer.writerow([
                    exp_name,
                    f"{np.mean(m['psnr']):.4f}", f"{np.std(m['psnr']):.4f}",
                    f"{np.mean(m['r2']):.6f}", f"{np.std(m['r2']):.6f}",
                    f"{np.mean(m['ssim']):.6f}", f"{np.std(m['ssim']):.6f}",
                    len(m['psnr'])
                ])
            writer.writerow([
                'OVERALL',
                f"{np.mean(all_psnr):.4f}", f"{np.std(all_psnr):.4f}",
                f"{np.mean(all_r2):.6f}", f"{np.std(all_r2):.6f}",
                f"{np.mean(all_ssim):.6f}", f"{np.std(all_ssim):.6f}",
                len(all_psnr)
            ])
        print(f"\n  Saved to {outpath}")
        
        # 儲存逐筆數據
        filename_detail = f'psnr_r2_ssim_detail_iter{it}.csv'
        outpath_detail = os.path.join(eval_dir, filename_detail)
        with open(outpath_detail, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['index', 'experiment', 'psnr', 'r2', 'ssim'])
            for i, pair in enumerate(pairs):
                exp_name = pair['exp_name']
                m = metrics_by_exp[exp_name]
                idx_in_exp = len([p for p in pairs[:i] if p['exp_name'] == exp_name])
                writer.writerow([
                    i, exp_name,
                    f"{m['psnr'][idx_in_exp]:.4f}",
                    f"{m['r2'][idx_in_exp]:.6f}",
                    f"{m['ssim'][idx_in_exp]:.6f}"
                ])
        print(f"  Saved detail data to {outpath_detail}")
        
        # ========================================================
        # 繪製視覺化對比圖
        # ========================================================
        print("\n  Generating comparison plots...")
        compare_dir = os.path.join(eval_dir, 'comparisons')
        os.makedirs(compare_dir, exist_ok=True)
        
        # 選取每個實驗的前 4 張做對比
        for exp_name in sorted(metrics_by_exp.keys()):
            exp_pairs = [p for p in pairs if p['exp_name'] == exp_name]
            n_show = min(4, len(exp_pairs))
            
            fig, axes = plt.subplots(2, n_show, figsize=(4*n_show, 8))
            fig.suptitle(f'{exp_name} — Real vs Generated', fontsize=16)
            
            for j in range(n_show):
                real_img = exp_pairs[j]['real'].permute(1, 2, 0).numpy()  # [H, W, 3]
                fake_img = exp_pairs[j]['fake'].permute(1, 2, 0).numpy()
                
                psnr_j = compute_psnr(exp_pairs[j]['fake'], exp_pairs[j]['real'])
                r2_j = compute_r_squared(exp_pairs[j]['fake'], exp_pairs[j]['real'])
                
                if n_show == 1:
                    ax_real = axes[0]
                    ax_fake = axes[1]
                else:
                    ax_real = axes[0, j]
                    ax_fake = axes[1, j]
                
                ax_real.imshow(np.clip(real_img, 0, 1))
                ax_real.set_title('Real')
                ax_real.axis('off')
                
                ax_fake.imshow(np.clip(fake_img, 0, 1))
                ax_fake.set_title(f'Gen (PSNR={psnr_j:.1f}, R²={r2_j:.3f})')
                ax_fake.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(compare_dir, f'compare_{exp_name}.png'), dpi=150)
            plt.close()
        
        print(f"  Saved comparison plots to {compare_dir}/")
        
        # ========================================================
        # 繪製指標分佈圖
        # ========================================================
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # PSNR 分佈
        for exp_name in sorted(metrics_by_exp.keys()):
            axes[0].hist(metrics_by_exp[exp_name]['psnr'], alpha=0.5, label=exp_name, bins=20)
        axes[0].set_xlabel('PSNR (dB)')
        axes[0].set_ylabel('Count')
        axes[0].set_title('PSNR Distribution')
        axes[0].legend()
        
        # R² 分佈
        for exp_name in sorted(metrics_by_exp.keys()):
            axes[1].hist(metrics_by_exp[exp_name]['r2'], alpha=0.5, label=exp_name, bins=20)
        axes[1].set_xlabel('R²')
        axes[1].set_ylabel('Count')
        axes[1].set_title('R² Distribution')
        axes[1].legend()
        
        # SSIM 分佈
        for exp_name in sorted(metrics_by_exp.keys()):
            axes[2].hist(metrics_by_exp[exp_name]['ssim'], alpha=0.5, label=exp_name, bins=20)
        axes[2].set_xlabel('SSIM')
        axes[2].set_ylabel('Count')
        axes[2].set_title('SSIM Distribution')
        axes[2].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(eval_dir, f'metrics_distribution_iter{it}.png'), dpi=150)
        plt.close()
        print(f"  Saved distribution plot to {eval_dir}/metrics_distribution_iter{it}.png")
    
    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60)