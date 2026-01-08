"""
调试脚本：注入监控代码来追踪 NerfAcc 的实际执行情况
运行方式: python debug_nerfacc.py
然后再运行训练
"""
import sys

# 计数器
call_counts = {
    'render_nerfacc': 0,
    'estimator_sampling': 0,
    'render': 0,
}

def inject_monitoring():
    """在 run_nerf_mod.py 中注入监控代码"""

    print("注入调试代码...")

    with open('submodules/nerf_pytorch/run_nerf_mod.py', 'r') as f:
        content = f.read()

    # 检查是否已经注入
    if '# DEBUG_INJECTION' in content:
        print("✓ 调试代码已存在，跳过注入")
        return

    # 在 render_nerfacc 开头添加计数
    injection1 = '''# DEBUG_INJECTION
    global _nerfacc_call_count
    if '_nerfacc_call_count' not in globals():
        _nerfacc_call_count = 0
    _nerfacc_call_count += 1
    if _nerfacc_call_count <= 5:
        print(f"[DEBUG] render_nerfacc called (#{_nerfacc_call_count}), bs={features.shape[0] if features is not None else 1}, N_rays={rays[0].shape[0] if rays is not None else 0}")
'''

    # 在 estimator.sampling 之前添加计数和计时
    injection2 = '''
    # DEBUG_INJECTION
    global _sampling_call_count
    if '_sampling_call_count' not in globals():
        _sampling_call_count = 0
    _sampling_call_count += 1

    import time
    if _sampling_call_count <= 5:
        print(f"[DEBUG] estimator.sampling called (#{_sampling_call_count}), rays_o.shape={rays_o.shape}")
    torch.cuda.synchronize()
    _sampling_start = time.time()
    # END DEBUG_INJECTION
'''

    injection3 = '''
    # DEBUG_INJECTION
    torch.cuda.synchronize()
    _sampling_time = (time.time() - _sampling_start) * 1000
    if _sampling_call_count <= 5:
        print(f"[DEBUG] estimator.sampling took {_sampling_time:.2f}ms, returned {len(t_starts_all):,} samples")
    # END DEBUG_INJECTION
'''

    # 替换
    # 1. 在 render_nerfacc 函数开头
    content = content.replace(
        '    if estimator is None:\n        raise ValueError("NerfAcc render requires an OccGridEstimator")',
        injection1 + '    if estimator is None:\n        raise ValueError("NerfAcc render requires an OccGridEstimator")'
    )

    # 2. 在 estimator.sampling 之前
    content = content.replace(
        '    # ⭐ 只調用一次 estimator.sampling()（之前是調用 bs=8 次！）\n    with torch.no_grad():',
        '    # ⭐ 只調用一次 estimator.sampling()（之前是調用 bs=8 次！）' + injection2 + '    with torch.no_grad():'
    )

    # 3. 在 estimator.sampling 之后
    content = content.replace(
        '        )\n\n    # 準備輸出',
        '        )' + injection3 + '\n    # 準備輸出'
    )

    # 在 render 函数开头添加计数
    injection4 = '''# DEBUG_INJECTION
    global _render_call_count
    if '_render_call_count' not in globals():
        _render_call_count = 0
    _render_call_count += 1
    if _render_call_count <= 5:
        print(f"[DEBUG] render (original) called (#{_render_call_count})")
'''

    content = content.replace(
        '    """原本的渲染函數（不使用 NerfAcc）"""\n    \n    if c2w is not None:',
        '"""原本的渲染函數（不使用 NerfAcc）"""' + injection4 + '    \n    if c2w is not None:'
    )

    # 写回文件
    with open('submodules/nerf_pytorch/run_nerf_mod.py', 'w') as f:
        f.write(content)

    print("✓ 调试代码注入完成")
    print("\n现在运行训练，前5次调用会显示调试信息:")
    print("  python train.py --config configs/nerfacc_optimized.yaml")


def remove_monitoring():
    """移除监控代码"""

    print("移除调试代码...")

    with open('submodules/nerf_pytorch/run_nerf_mod.py', 'r') as f:
        lines = f.readlines()

    # 移除所有包含 DEBUG_INJECTION 的行
    filtered_lines = []
    skip_until_end = False

    for line in lines:
        if '# DEBUG_INJECTION' in line:
            skip_until_end = True
            continue
        elif skip_until_end and '# END DEBUG_INJECTION' in line:
            skip_until_end = False
            continue
        elif not skip_until_end:
            filtered_lines.append(line)

    with open('submodules/nerf_pytorch/run_nerf_mod.py', 'w') as f:
        f.writelines(filtered_lines)

    print("✓ 调试代码已移除")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'remove':
        remove_monitoring()
    else:
        inject_monitoring()
        print("\n要移除调试代码，运行:")
        print("  python debug_nerfacc.py remove")