"""
Cohen's Kappa Calculation Demo
===============================
A complete, runnable demonstration of how to compute inter-annotator
agreement (Cohen's Kappa) for object detection annotation.

This is the most critical metric for label quality in real-world
annotation projects.

Usage:
    python assets/figures/11-1-cohen-kappa-demo.py
"""
import numpy as np


def cohens_kappa(y_a, y_b):
    """Compute Cohen's Kappa for two annotators."""
    y_a = np.asarray(y_a)
    y_b = np.asarray(y_b)
    n = len(y_a)
    assert len(y_b) == n, "Length mismatch"

    classes = np.unique(np.concatenate([y_a, y_b]))
    k = len(classes)
    cm = np.zeros((k, k), dtype=int)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    for a, b in zip(y_a, y_b):
        cm[class_to_idx[a], class_to_idx[b]] += 1

    p_o = np.trace(cm) / n
    row_sums = cm.sum(axis=1) / n
    col_sums = cm.sum(axis=0) / n
    p_e = (row_sums * col_sums).sum()
    kappa = (p_o - p_e) / (1 - p_e + 1e-12)
    return kappa, p_o, p_e, cm


def main():
    print("=" * 60)
    print("Cohen's Kappa: Inter-Annotator Agreement Demo")
    print("=" * 60)

    # Demo 1: 完美的标注员
    print("\n[Demo 1] 完美一致的两个标注员")
    y_a = ['cat', 'dog', 'car', 'cat', 'dog', 'car'] * 10
    y_b = y_a.copy()
    kappa, p_o, p_e, _ = cohens_kappa(y_a, y_b)
    print(f"   κ = {kappa:.4f} (p_o={p_o:.3f}, p_e={p_e:.3f})")
    print(f"   → 解读: 完美一致 ✓")

    # Demo 2: 高质量工业标注
    print("\n[Demo 2] 高质量标注 (10% 争议)")
    y_a = np.array(['cat']*100 + ['dog']*100 + ['car']*100)
    rng = np.random.RandomState(42)
    y_b = y_a.copy()
    # 10% 噪声
    for cls in ['cat', 'dog', 'car']:
        idx = np.where(y_a == cls)[0]
        flip = rng.choice(idx, size=10, replace=False)
        other = [c for c in ['cat', 'dog', 'car'] if c != cls]
        for fi in flip:
            y_b[fi] = rng.choice(other)
    kappa, p_o, p_e, _ = cohens_kappa(y_a, y_b)
    print(f"   κ = {kappa:.4f} (p_o={p_o:.3f}, p_e={p_e:.3f})")
    print(f"   → 解读: 实质性一致 (Moderate), 工业可用 ✓")

    # Demo 3: 低质量标注 (类别定义不清)
    print("\n[Demo 3] 低质量标注 (类别定义不清, 30% 争议)")
    y_a = np.array(['cat']*100 + ['dog']*100 + ['car']*100)
    rng = np.random.RandomState(0)
    y_b = y_a.copy()
    for cls in ['cat', 'dog', 'car']:
        idx = np.where(y_a == cls)[0]
        flip = rng.choice(idx, size=30, replace=False)
        other = [c for c in ['cat', 'dog', 'car'] if c != cls]
        for fi in flip:
            y_b[fi] = rng.choice(other)
    kappa, p_o, p_e, _ = cohens_kappa(y_a, y_b)
    print(f"   κ = {kappa:.4f} (p_o={p_o:.3f}, p_e={p_e:.3f})")
    print(f"   → 解读: 一般 (Slight), 需要重新定义类别并培训标注员 ⚠")

    # Demo 4: 工业阈值检查
    print("\n[Demo 4] 工业阈值检查 (target κ ≥ 0.7)")
    target = 0.7
    test_kappas = [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35]
    for k in test_kappas:
        status = "✓ 通过" if k >= target else "✗ 需改进"
        print(f"   κ = {k:.2f}  {status}")

    # 类别不平衡的修正提醒
    print("\n[提示] 类别严重不平衡时, 用 Quadratic Weighted Kappa 更稳健")
    print("       或 报告 Per-Class F1 分数 + Confusion Matrix")


if __name__ == "__main__":
    main()
