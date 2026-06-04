"""
Training Hardware Benchmark & Cost Calculator
==============================================
A runnable Python demo that calculates the cost-effectiveness of
different hardware configurations for a given training task.

Usage:
    python assets/figures/10-8-cost-calculator.py
"""
import time
import numpy as np


def benchmark_pretend(gpu_name, fp16_tflops, vram_gb, price_usd):
    """Simulate a training run: ResNet-50 on ImageNet, batch 256.
    Returns: hours_to_train, cost_per_run, cost_efficiency
    """
    # ImageNet ResNet-50 baseline: ~30 GPU-hours on A100
    # Reference: A100 = 77.9 TFLOPS = 30 hours
    baseline_hours = 30
    baseline_tflops = 77.9  # A100 FP16
    hours = baseline_hours * (baseline_tflops / fp16_tflops)

    # Apply multi-GPU efficiency: 4 GPU = ~3.5x speedup
    # (already accounted for by the 1.5 scaling here)
    if price_usd >= 10000:  # datacenter
        hours *= 1.5

    cost_per_hour = price_usd / 10000  # amortization over ~10000 hours
    cost_per_run = cost_per_hour * hours

    return {
        'gpu': gpu_name,
        'vram': vram_gb,
        'hours': hours,
        'cost_per_run': cost_per_run,
        'cost_per_hour_amortized': cost_per_hour,
        'tflops_per_dollar': fp16_tflops / price_usd * 1000,
    }


def main():
    print("=" * 70)
    print("AI 训练硬件成本计算器 (ResNet-50 ImageNet 训练基准)")
    print("=" * 70)
    print("\n注: 这是基于 A100 = 30 GPU-hours 的简化估算")
    print("    实际时间受 batch size / 优化器 / 网络结构影响 ±30%\n")

    gpus = [
        ('RTX 4060',    15.1,  8,   300),
        ('RTX 4070',    29.1,  12,  600),
        ('RTX 4090',    82.6,  24,  1600),
        ('A5000',       77.4,  24,  1700),
        ('A6000 Ada',   91.1,  48,  6800),
        ('A100 80G',    77.9,  80,  15000),
        ('H100 80G',    197.9, 80,  30000),
    ]

    results = [benchmark_pretend(*g) for g in gpus]

    print(f"{'GPU':<14} {'VRAM':<6} {'小时':<8} {'$/次':<8} {'$/时摊销':<10} {'TFLOPS/$':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['gpu']:<14} {r['vram']:<4}GB {r['hours']:<8.1f} "
              f"${r['cost_per_run']:<7.0f} ${r['cost_per_hour_amortized']:<9.2f} "
              f"{r['tflops_per_dollar']:<10.2f}")

    # 性价比冠军
    best_value = min(results, key=lambda r: r['cost_per_run'])
    fastest = min(results, key=lambda r: r['hours'])
    print(f"\n🏆 性价比冠军: {best_value['gpu']} (${best_value['cost_per_run']:.0f}/次训练)")
    print(f"⚡ 速度冠军: {fastest['gpu']} ({fastest['hours']:.1f} 小时/次训练)")

    # LLM 训练对比
    print("\n" + "=" * 70)
    print("LLM 7B 全参微调 (假设需要 ~600 GPU-hours on H100)")
    print("=" * 70)
    print(f"\n{'GPU':<14} {'小时':<10} {'$/次':<12} {'可行性':<10}")
    print("-" * 70)
    llm_baseline_hours = 600
    for r in results:
        scale = 77.9 / next(g[1] for g in gpus if g[0] == r['gpu'])
        hours = llm_baseline_hours * scale
        if r['vram'] < 16:
            viability = "❌ 显存不足"
        elif r['vram'] < 24:
            viability = "⚠️ 极慢/需优化"
        else:
            viability = "✓ 可行"
        cost = r['cost_per_hour_amortized'] * hours
        print(f"{r['gpu']:<14} {hours:<10.0f} ${cost:<11.0f} {viability:<10}")


if __name__ == "__main__":
    main()
