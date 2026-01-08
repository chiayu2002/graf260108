"""
简单的 NerfAcc 速度测试脚本
直接测试渲染速度，不涉及完整训练
"""
import torch
import time
import sys
import numpy as np

# 添加路径
sys.path.insert(0, '.')

from graf.config import load_config, build_models

def test_rendering_speed(config_path, n_iterations=100):
    """测试渲染速度"""

    print("="*70)
    print("NerfAcc 渲染速度测试")
    print("="*70)

    # 加载配置
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 构建模型
    generator, _ = build_models(config, disc=False)
    generator = generator.to(device)
    generator.eval()

    # 准备输入
    zdist = torch.distributions.normal.Normal(
        torch.zeros(config['z_dist']['dim']).to(device),
        torch.ones(config['z_dist']['dim']).to(device)
    )

    num_classes = config['discriminator']['num_classes']
    label = torch.zeros(1, num_classes).to(device)

    # 检查 NerfAcc 状态
    use_nerfacc = hasattr(generator, 'use_nerfacc') and generator.use_nerfacc
    print(f"\nNerfAcc 状态: {'启用' if use_nerfacc else '禁用'}")

    if use_nerfacc:
        print(f"Grid 解析度: {config['nerfacc'].get('resolution', 128)}")
        print(f"Render step size: {generator.render_step_size}")

    print(f"\n开始测试 {n_iterations} 次渲染...")
    print("-"*70)

    # 预热
    for _ in range(5):
        z = zdist.sample((1,))
        with torch.no_grad():
            _ = generator(z, label)

    torch.cuda.synchronize()

    # 正式测试
    render_times = []

    for i in range(n_iterations):
        z = zdist.sample((1,))

        torch.cuda.synchronize()
        start = time.time()

        with torch.no_grad():
            fake_img = generator(z, label)

        torch.cuda.synchronize()
        render_time = (time.time() - start) * 1000  # ms

        render_times.append(render_time)

        if (i + 1) % 10 == 0:
            avg_time = np.mean(render_times[-10:])
            print(f"[{i+1:3d}/{n_iterations}] 最近10次平均: {avg_time:.2f} ms ({1000/avg_time:.1f} renders/sec)")

    # 统计
    print("\n" + "="*70)
    print("测试结果")
    print("="*70)

    avg_time = np.mean(render_times)
    std_time = np.std(render_times)
    min_time = np.min(render_times)
    max_time = np.max(render_times)

    print(f"\n渲染时间统计:")
    print(f"  平均: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"  最小: {min_time:.2f} ms")
    print(f"  最大: {max_time:.2f} ms")

    print(f"\n渲染吞吐量:")
    renders_per_sec = 1000 / avg_time
    renders_per_min = renders_per_sec * 60
    print(f"  {renders_per_sec:.1f} renders/sec")
    print(f"  {renders_per_min:.1f} renders/min")

    # 如果有 extras 信息
    if hasattr(generator, 'last_render_extras') and generator.last_render_extras:
        extras = generator.last_render_extras
        if 'n_samples' in extras:
            print(f"\n采样统计:")
            print(f"  实际采样点: {extras['n_samples']:,}")
            if 'theoretical_samples' in extras:
                print(f"  理论采样点: {extras['theoretical_samples']:,}")
                print(f"  采样减少率: {extras.get('sample_reduction', 0):.1f}%")

    print("\n" + "="*70)

    return avg_time, renders_per_min


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/nerfacc_optimized.yaml')
    parser.add_argument('--iters', type=int, default=100)
    args = parser.parse_args()

    test_rendering_speed(args.config, args.iters)