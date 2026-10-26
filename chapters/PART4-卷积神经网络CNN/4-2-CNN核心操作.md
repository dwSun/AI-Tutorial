# 第4-2章 CNN 核心操作

## 学习目标

- 掌握池化层的两种主要类型：最大池化（Max Pooling）和平均池化（Average Pooling）
- 理解池化层的降维、防止过拟合的作用
- 理解 Batch Normalization 在 CNN 中的应用
- 理解全连接层在 CNN 中的作用
- 能够描述一个完整 CNN 的架构示例

---

## 1. 池化层（Pooling Layer）

### 1.1 什么是池化

池化（Pooling）也称为**降采样（Downsampling）**，是将输入图像某个矩形区域的所有数值汇总成一个代表性数值的操作。

与卷积类似，池化也有：
- **滤波器尺寸（Filter Size）**：如 2×2、3×3
- **步幅（Stride）**：通常等于滤波器尺寸（不重叠）
- **填充（Padding）**：通常为 0

### 1.2 最大池化（Max Pooling）

在感受野内取**最大值**输出。

```
最大池化 2×2, stride=2:

输入:                      输出:

[ 1  3  2  1 ]             [ 7  3 ]
[ 6  7  8  2 ]      →      [ 9  6 ]
[ 4  2  1  0 ]
[ 9  5  3  6 ]

每个 2×2 窗口取最大值:
[1,3]→3  [2,1]→3      →   [7, 3]
[6,7]→7  [8,2]→8             [9, 6]
[4,2]→4  [1,0]→1
[9,5]→9  [3,6]→6
```

### 1.3 平均池化（Average Pooling）

在感受野内取**平均值**输出。

```
平均池化 2×2, stride=2:

输入:                      输出:

[ 1  3  2  1 ]             [ 4.25  2.00 ]
[ 6  7  8  2 ]      →      [ 4.75  2.50 ]
[ 4  2  1  0 ]
[ 9  5  3  6 ]

每个 2×2 窗口求平均:
(1+3+6+7)/4=4.25  (2+1+8+2)/4=3.25  →  [4.25, 3.25]
(4+2+9+5)/4=5.00  (1+0+3+6)/4=2.50       [5.00, 2.50]
```

### 1.4 池化的作用

**1. 降维（减少参数数量）**

```
输入 4×4 = 16 个值 → 输出 2×2 = 4 个值
压缩比 4:1，大幅减少后续层计算量
```

**2. 防止过拟合**

- 池化后的特征对微小位移不那么敏感
- 形成"特征竞争"：只有最强的特征被保留
- 相当于一种粗糙的信息压缩

**3. 增大感受野**

- 池化让后续卷积层能"看到"更大的输入区域
- 同样的卷积核，在池化后的特征图上操作，感受野更广

**4. 对输入尺度变化的容忍度**

```
输入 224×224 → Conv → 112×112 → Pool → 56×56
输入 192×192 → Conv → 96×96   → Pool → 48×48

无论输入多大，经过多层池化后特征图尺寸都归一化了
最后一层可以用 Global Average Pooling 处理任意尺度输入
```

### 1.5 池化类型对比

| 类型 | 公式 | 特点 |
|------|------|------|
| Max Pooling | `output = max(window)` | 保留最显著的特征，对纹理提取有效 |
| Average Pooling | `output = mean(window)` | 保留背景信息，平滑特征 |
| Global Average Pooling | `output = mean(整个特征图)` | 替代全连接层，减少参数 |

---

## 2. Batch Normalization（批量归一化）

### 2.1 什么是 Batch Normalization

Batch Normalization（简称 BN）是一种正则化技术，在每一层的输入上做归一化处理，使其均值为 0、方差为 1。

```
计算过程（以单个神经元为例）：
1. 计算 mini-batch 内该神经元的均值 μ_B
2. 计算 mini-batch 内该神经元的方差 σ²_B
3. 归一化: x̂ = (x - μ_B) / √(σ²_B + ε)
4. 线性变换: y = γ·x̂ + β    （γ,β 为可学习参数）
```

### 2.2 为什么 CNN 需要 BN

在 CNN 中，网络的层数很深，前面层参数的变化会逐层放大，导致：

- **Internal Covariate Shift**：浅层参数变化导致深层输入分布不断变化
- **梯度消失/爆炸**：激活函数在极端值处饱和
- **训练不稳定**：需要非常小的学习率

BN 通过对每一层的输入归一化，有效缓解上述问题。

### 2.3 CNN 中的 BN 应用

```
卷积层 → BatchNorm → ReLU → 池化层 → ...
```

在 CNN 中，BN 通常应用在卷积层之后、激活函数之前：

```
Conv2d → BatchNorm2d → ReLU → Pool → ...
```

> 注意：BN 在测试时使用的是训练过程中学到的全局统计量（running mean/variance），而非 batch 统计量。

### 2.4 BN 的优势

| 优势 | 说明 |
|------|------|
| 加速收敛 | 允许使用更大的学习率 |
| 减轻梯度消失 | 归一化让激活函数工作在天平中央 |
| 正则化 | 每个 batch 的随机性有轻微正则化效果 |
| 减少对 Dropout 的依赖 | BN 本身有一定正则化作用 |

---

## 3. 五种归一化方法（BN/LN/IN/GN/SN）

归一化（Normalization）是深度学习的"稳定器"。不同归一化方法**在哪个维度上做归一化**决定了它们的适用场景。

### 3.1 归一化维度对比

设输入特征 $x$ 形状为 $(N, C, H, W)$，其中 $N$ 是 batch，$C$ 是通道，$H, W$ 是空间维度。

| 方法 | 归一化维度 | 形状 | 适用场景 |
|------|-----------|------|----------|
| **BN (BatchNorm)** | 跨 batch + 跨 H, W | $(N, H, W) \to C$ 个统计量 | CNN 大 batch 训练 |
| **LN (LayerNorm)** | 跨 C + 跨 H, W | $(C, H, W) \to N$ 个统计量 | Transformer、NLP |
| **IN (InstanceNorm)** | 跨 H, W | $(H, W) \to N \times C$ 个统计量 | 风格迁移 |
| **GN (GroupNorm)** | 跨 group 内 C + 跨 H, W | 分组内归一化 | 检测/分割小 batch |
| **SN (SwitchableNorm)** | 学习加权三种归一化 | 动态权重 | 通用 |

### 3.2 Batch Normalization（BN, 2015）

**详见第 2 节**。BN 是 CNN 训练的事实标准。

```python
nn.BatchNorm2d(num_features, eps=1e-5, momentum=0.1, affine=True)
```

**核心限制**：
- 依赖 batch 统计量
- 训练时 batch size ≥ 16 才有意义
- 推理时用 running mean/var

### 3.3 Layer Normalization（LN, 2016）

**对单个样本**的所有通道和空间维度做归一化：

$$
\mu_n = \frac{1}{CHW} \sum_{c,h,w} x_{n,c,h,w}
$$

$$
\sigma_n^2 = \frac{1}{CHW} \sum_{c,h,w} (x_{n,c,h,w} - \mu_n)^2
$$

**特点**：
- 不依赖 batch
- 训练/推理行为一致
- 是 **Transformer 的标配**（与 BN 不同，NLP 中 LN 比 BN 效果更好）

```python
nn.LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True)

# 例：Transformer block
class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_ff):
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(),
            nn.Linear(dim_ff, d_model)
        )
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ff(self.norm2(x))
        return x
```

### 3.4 Instance Normalization（IN, 2016）

**对每个样本的每个通道**做归一化：

$$
\mu_{n,c} = \frac{1}{HW} \sum_{h,w} x_{n,c,h,w}, \quad \sigma_{n,c}^2 = \frac{1}{HW} \sum_{h,w} (x_{n,c,h,w} - \mu_{n,c})^2
$$

**特点**：
- 等价于 BN 但 batch=1
- 适合**风格迁移**任务（去除每张图的风格差异）

```python
nn.InstanceNorm2d(num_features, eps=1e-5, affine=False)
```

### 3.5 Group Normalization（GN, ECCV 2018）

**将通道分成 G 组，每组内做 LN**：

```mermaid
flowchart TD
  A[输入 N×C×H×W] --> B[切分为 G 组]
  B --> C1[Group 1<br/>C/G 通道]
  B --> C2[Group 2<br/>C/G 通道]
  B --> C3[Group 3<br/>C/G 通道]
  C1 --> D1[组内 LN 归一化]
  C2 --> D2[组内 LN 归一化]
  C3 --> D3[组内 LN 归一化]
  D1 --> E[Concat 输出]
  D2 --> E
  D3 --> E
```

**特点**：
- **不依赖 batch size**
- 在 batch=2~4 时也稳定
- **检测/分割任务的标配**（因 batch 通常小）

```python
# num_groups 常见 32
nn.GroupNorm(num_groups=32, num_channels=512, eps=1e-5)
```

**性能对比**（ResNet-50, ImageNet）：

| Batch Size | BN Top-1 | GN Top-1 |
|-----------|----------|----------|
| 32 | 76.0% | 75.9% |
| 16 | 75.8% | 76.0% |
| 8 | 75.0% | 76.1% |
| 4 | 72.0% | **75.9%** |
| 2 | 65.0% | **75.0%** |
| 1 | 56.0% | **74.0%** |

GN 在小 batch 下优势明显！

### 3.6 Switchable Normalization（SN, ICLR 2018）

**用可学习权重**混合 BN + IN + LN：

$$
\hat{x}_{n,c,h,w} = \frac{x_{n,c,h,w} - \sum_{k \in \{BN, IN, LN\}} \lambda_k \mu_k}{\sqrt{\sum_{k} \lambda_k' \sigma_k^2 + \epsilon}} \cdot \gamma + \beta
$$

其中 $\lambda_k$ 由 softmax 控制。

**特点**：
- 通用性强
- 适合 batch 不固定的场景（如检测）
- 计算量稍大

```python
# 通过 mmcv 或自实现
import mmcv
norm = mmcv.norm.SwitchableNorm(num_features, num_groups=1)
```

### 3.7 五种归一化对比图

### 3.8 选型决策

| 任务 | 归一化 | 理由 |
|------|--------|------|
| CNN 分类（batch≥16） | BN | 标准选择 |
| Transformer / NLP | LN | 与 batch 无关 |
| 风格迁移 | IN | 单图风格归一 |
| 检测/分割（小 batch） | GN | 替代 BN |
| 检测（mmdet 默认） | SyncBN | 跨卡同步 BN 统计量 |
| 大模型 | LN/RMSNorm | 稳定性更好 |

### 3.9 SyncBN（同步 BN）

跨多卡同步 BN 统计量，解决**分布式训练中 batch 太小**的问题。

```python
# mmcv 实现
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = build_model(...)
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
```

---

## 4. 全连接层（Fully Connected Layer）

### 4.1 什么是全连接层

全连接层（FC 层）的每个神经元与前一层的所有神经元都相连。在 CNN 中通常出现在最后几层，用于将学到的特征映射到最终的分类/回归输出。

```
全连接层示意：

特征图 (7×7×512)
    ↓ 展平
25088 个神经元
    ↓ 全连接
4096 个神经元
    ↓ 全连接
1000 个类别 (Softmax)
```

### 4.2 全连接层在 CNN 中的位置

```
典型 CNN 架构末尾:

... → Conv → Conv → Pool → Conv → Conv → Pool →
GlobalAvgPool → FC → FC → Softmax → Output
```

- **特征提取阶段**：多个卷积层 + 池化层，逐层提取从边缘到语义的特征
- **分类阶段**：1~2 个全连接层，将特征映射到类别空间

### 4.3 Global Average Pooling（GAP）

GAP 是用全局平均池化替代全连接层的方案：

```
特征图 7×7×1000
    ↓ GAP
1×1×1000
    ↓ 直接对应 1000 类
每个通道对应一个类别概率
```

**GAP 的优势**：
- 大幅减少参数数量（无权重矩阵）
- 对输入尺度不再敏感
- 减少过拟合风险

> GoogleNet、VGG 等经典网络的分类头都使用 GAP。

---

## 5. 完整 CNN 架构示例

### 5.1 LeNet-5 完整架构

```
输入: 32×32×1 (灰度手写数字)

Conv1: 6 个 5×5 卷积核, stride=1 → 28×28×6
Pool1: 2×2 Max Pool, stride=2    → 14×14×6

Conv2: 16 个 5×5 卷积核, stride=1 → 10×10×16
Pool2: 2×2 Max Pool, stride=2    → 5×5×16

FC1: 120 个神经元
FC2: 84 个神经元
Output: 10 类 (Softmax)
```

### 5.2 一个典型图像分类 CNN 的结构规律

```
输入图像
    ↓
特征提取阶段（浅层 → 深层）
Conv × N → BN → ReLU → Pool
    ↑__________________|
    （重复多次，通道数逐层增加）
    ↓
全局特征聚合
Global Average Pooling
    ↓
分类头
FC (可选) → Softmax → 类别概率
```

| 阶段 | 作用 | 特征类型 |
|------|------|----------|
| 浅层 Conv | 提取边缘、纹理、颜色斑块 | 局部、低级 |
| 中层 Conv | 组合低级特征，形成部件 | 中级 |
| 深层 Conv | 提取完整物体语义特征 | 全局、高级 |
| GAP/FC | 聚合特征，映射到类别空间 | 分类决策 |

### 5.3 经典 CNN 各层通道数变化趋势

```
以 VGG-16 为例（简化）:

Input: 224×224×3

Conv1_1: 64 个 3×3  → 224×224×64
Conv1_2: 64 个 3×3  → 224×224×64
Pool1: 2×2 stride=2 → 112×112×64

Conv2_1: 128 个     → 112×112×128
Conv2_2: 128 个     → 112×112×128
Pool2:              → 56×56×128

Conv3_1: 256 个     → 56×56×256
Conv3_2: 256 个     → 56×56×256
Conv3_3: 256 个     → 56×56×256
Pool3:              → 28×28×256

Conv4_1: 512 个     → 28×28×512
...（通道数 512）
Pool4:              → 14×14×512

Conv5_1: 512 个     → 14×14×512
...（通道数 512）
Pool5:              → 7×7×512

GAP:                → 1×1×512
FC: 1000 类         → 输出概率

通道数: 64 → 128 → 256 → 512（逐层翻倍）
特征图尺寸: 224 → 112 → 56 → 28 → 14 → 7（逐层减半）
```

---

## 6. 总结

```
┌──────────────────────────────────────────────────────┐
│                 CNN 核心操作总结                     │
├──────────────────────────────────────────────────────┤
│ 卷积层:   提取局部特征，参数共享                     │
│ 池化层:   降维、抗干扰、增加感受野                   │
│ BN层:     稳定训练、加速收敛、正则化                 │
│ FC层:     特征映射到分类空间（可用 GAP 替代）        │
└──────────────────────────────────────────────────────┘
```

| 操作 | 位置 | 作用 |
|------|------|------|
| Conv + BN + ReLU | 特征提取核心 | 提取特征 + 归一化 + 非线性激活 |
| Pooling | 周期插入（每1-3层Conv后） | 降维、增加感受野、抗过拟合 |
| GAP/FC | 分类头前 | 聚合全局特征、映射到类别 |
| Softmax | 最后一层 | 多分类概率输出 |

---

## Python 代码示例：CNN核心操作可视化

下面通过 Python 代码演示 CNN 核心操作的效果：

![CNN 核心操作可视化（池化 / BN / GAP / 架构）](../../assets/figures/4-2-cnn-core-operations.png)

> 绘图代码：[`assets/figures/4-2-cnn-core-operations.py`](../../assets/figures/4-2-cnn-core-operations.py)

**关键结论**：
- **池化**：Max Pooling 保留最强响应，Average Pooling 平滑特征
- **BN**：将各层激活值分布拉回 N(0, 1)，加速收敛、缓解梯度问题
- **GAP vs FC**：7×7×512 → 1000 类，GAP 只需 0.5M 参数，FC 需 ~119M（减少 99.6%）
- **架构尺寸**：随网络加深，空间尺寸减半、通道数翻倍（典型 CNN 设计模式）

### 代码说明

1. **池化操作可视化**：
   - 展示了4×4特征图上的2×2 Max Pooling和Average Pooling操作
   - Max Pooling保留最显著特征，Average Pooling平滑特征
   - 图中标注了每个池化区域的来源

2. **Batch Normalization效果**：
   - 模拟了不同层激活值的分布
   - 展示了BN如何将各层激活值归一化为均值0、方差1的分布
   - 减少了内部协变量偏移，加速训练收敛

3. **GAP vs FC参数对比**：
   - Global Average Pooling大幅减少参数数量
   - 示例：7×7×512特征图 → 1000类分类
   - GAP: 512×1000 ≈ 0.5M参数
   - FC: 7×7×512×4096+4096×4096+4096×1000 ≈ 118M参数
   - GAP减少99.6%参数

4. **CNN架构特征图变化**：
   - 展示了典型CNN架构中特征图尺寸和通道数的变化规律
   - 特征图尺寸逐层减半（224→112→56→28）
   - 通道数逐层翻倍（64→128→256）

通过运行这段代码，读者可以直观理解CNN核心操作的作用和效果。

---

## 本章小结

1. **池化层（Pooling）**：降采样操作，将矩形区域汇总为一个值。
   - **Max Pooling**：取最大值，保留最显著特征，对纹理提取有效
   - **Average Pooling**：取平均值，保留背景信息，平滑特征
   - **作用**：降维、防止过拟合、增大感受野、对输入尺度变化容忍度高

2. **Batch Normalization（BN）**：对每一层输入做归一化（均值 0、方差 1），再线性变换。
   - **作用**：加速收敛、减轻梯度消失、有正则化效果
   - **位置**：卷积层 → BN → ReLU → 池化层
   - **测试时**：使用训练时学到的全局统计量

3. **全连接层（FC）**：将特征映射到分类空间。传统 CNN 末尾用 FC 做分类决策，可用 Global Average Pooling（GAP）替代以减少参数。

4. **典型 CNN 架构规律**：Conv×N → BN → ReLU → Pool（重复）→ GAP → FC → Softmax。通道数逐层翻倍（64→128→256→512），特征图尺寸逐层减半。

---


## 知识图谱

```mermaid
mindmap
  root((第4-2章 CNN 核心操作))
    学习目标
    1. 池化层（Pooling Layer）
      1.1 什么是池化
      1.2 最大池化（Max Pooling）
      1.3 平均池化（Average Pooling）
      1.4 池化的作用
    2. Batch Normalization（批量归一化）
      2.1 什么是 Batch Normalizati
      2.2 为什么 CNN 需要 BN
      2.3 CNN 中的 BN 应用
      2.4 BN 的优势
    3. 五种归一化方法（BN/LN/IN/GN/SN）
      3.0 归一化维度对比
      3.1 Batch Normalization（B
      3.2 Layer Normalization（L
      3.6 五种归一化对比图
    4. 全连接层（Fully Connected Layer）
      3.1 什么是全连接层
      3.2 全连接层在 CNN 中的位置
      3.3 Global Average Poolin
    5. 完整 CNN 架构示例
      4.1 LeNet-5 完整架构
      4.2 一个典型图像分类 CNN 的结构规律
      4.3 经典 CNN 各层通道数变化趋势
    Python 代码示例：CNN核心操作可视化
      代码说明
```

## 参考资料

1. LeCun, Y. et al. (1998). *Gradient-Based Learning Applied to Document Recognition* (LeNet-5)
2. Ioffe, S. & Szegedy, C. (2015). *Batch Normalization: Accelerating Deep Network Training*
3. Lin, M. et al. (2014). *Network In Network*
4. Simonyan, K. & Zisserman, A. (2015). *Very Deep Convolutional Networks for Large-Scale Image Recognition* (VGGNet)