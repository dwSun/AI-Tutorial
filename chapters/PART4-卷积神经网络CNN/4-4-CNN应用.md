# 第4-4章 CNN 应用

## 学习目标

- 理解 CNN 在图像分类任务中的应用，包括 ImageNet 竞赛的推动
- 了解目标检测的两阶段（R-CNN）和单阶段（YOLO）方法
- 了解语义分割的经典方法：FCN、U-Net、DeepLab
- 掌握迁移学习的概念和用预训练模型微调的方法
- 了解常用预训练模型库（torchvision、timm）

---

## 1. 图像分类（Image Classification）

### 1.1 ImageNet 与 CNN 的互相推动

ImageNet 是计算机视觉领域最大的图像分类数据集之一，包含 1000 个类别、约 140 万张标注图像。

ImageNet 竞赛（ILSVRC）是 CNN 发展的核心推动力：

```mermaid
flowchart LR
    A["2010 26.2%<br>传统方法"] --> B["2011 25.8%<br>传统方法改进"]
    B --> C["2012 16.4%<br>AlexNet首次参赛"]
    C --> D["2013 11.7%<br>ZFNet"]
    D --> E["2014 7.3%<br>VGG/GoogleNet超越人类"]
    E --> F["2015 3.6%<br>ResNet"]
    F --> G["2016+ <2.5%<br>深度网络饱和"]
```



> 人类在 ImageNet 上的分类错误率约为 5.1%，2014 年 CNN 首次超越人类！

### 1.2 图像分类的评价指标

| 指标 | 说明 |
|------|------|
| Top-1 准确率 | 预测概率最高的那一类正确 |
| Top-5 准确率 | 预测概率前 5 名中有正确类别 |
| 参数量 | 模型大小，影响内存和推理速度 |
| FLOPs | 浮点运算次数，衡量计算复杂度 |
| FPS | 每秒处理帧数，实时性指标 |

### 1.3 图像分类的典型流程

```mermaid
flowchart LR
    A[准备数据] --> B[数据增强]
    B --> C[构建模型]
    C --> D[训练]
    D --> E[验证]
    E --> F[测试]
    F --> G[部署]
```

---

## 2. 目标检测（Object Detection）

### 2.1 什么是目标检测

目标检测不仅要识别图像中的物体类别，还要**定位**物体位置（用边界框 Bounding Box 表示）。

**输出**：
- 类别标签（Class Label）
- 置信度分数（Confidence Score）
- 边界框坐标（Bounding Box: x, y, width, height）

### 2.2 两阶段检测器：R-CNN 系列

R-CNN 系列是"两阶段"检测器的代表：

**流程**：
1. **阶段1（区域提议）**：从图像中找出可能包含物体的候选区域（Region Proposal）
2. **阶段2（分类与回归）**：对每个候选区域进行分类和边界框精调

```mermaid
flowchart LR
    A[R-CNN 2014] -->|"区域提议→特征提取→SVM→边界框回归<br>慢，47秒/图"| B[Fast R-CNN 2015]
    B -->|"区域提议→全图特征→ROI Pooling→分类<br>共享卷积，2.3秒/图"| C[Faster R-CNN 2015]
    C -->|"区域提议网络RPN→全图特征→ROI Pooling→分类<br>端到端，0.2秒/图"| D[Mask R-CNN 2017]
    D -->|"Faster R-CNN + 实例分割<br>目标检测+实例分割"| E((演进))
```

### 2.3 单阶段检测器：YOLO 系列

YOLO（You Only Look Once）是"单阶段"检测器的代表，直接从图像预测物体位置和类别。

**核心思想**：将检测任务视为单一的回归问题，在一次前向传播中同时预测所有边界框和类别概率。

```mermaid
flowchart LR
    A[输入图像] --> B[划分网格<br>如 7×7]
    B --> C{每个网格预测}
    C -->|B个边界框| D[x, y, w, h, confidence]
    C -->|C个类别概率| E[类别概率]
    D --> F[合并所有预测]
    E --> F
    F --> G[NMS<br>非极大抑制]
    G --> H[最终检测结果]
    
    style H fill:#90EE90
```



**YOLO 版本演进**：

| 版本 | 年份 | 主要改进 |
|------|------|----------|
| YOLOv1 | 2016 | 首次单阶段检测，简单但对小物体效果差 |
| YOLOv2 | 2017 | 引入 Anchor Box、BN、Passthrough |
| YOLOv3 | 2018 | 多尺度预测（FPN）、Darknet-53 |
| YOLOv4 | 2020 | CSPDarknet、Mish、CIoU Loss |
| YOLOv5 | 2021 | 自动化调参、部署友好 |
| YOLOv7 | 2022 | 高效模型架构、E-ELAN |
| YOLOv8 | 2023 | 任务对齐、Anchor-free |
| YOLOv9 | 2024 | GELAN、PGI（Programmable Gradient Information） |

### 2.4 两阶段 vs 单阶段对比

| 对比维度 | 两阶段（R-CNN系列） | 单阶段（YOLO系列） |
|----------|---------------------|-------------------|
| 检测速度 | 较慢（1-10 FPS） | 快（30-150 FPS） |
| 检测精度 | 较高（尤其是小物体） | 稍低，但近年已接近 |
| 适用场景 | 对精度要求高、离线处理 | 实时性要求高 |
| 经典模型 | Faster R-CNN、Mask R-CNN | YOLOv5-v9、SSD |

---

## 3. 语义分割（Semantic Segmentation）

### 3.1 什么是语义分割

语义分割是对图像中每个像素进行类别预测，不区分同类物体的不同实例。

```mermaid
flowchart LR
    A[原始图像] --> B[分割结果]
    
    subgraph original_img[原始图像]
        C1["[ cat cat cat ]"]
        C2["[ cat dog cat ]"]
        C3["[ cat cat dog ]"]
    end
    
    subgraph seg_result[分割结果]
        D1["[ 1  1  1 ]"]
        D2["[ 1  2  1 ]"]
        D3["[ 1  1  2 ]"]
    end
    
    A --> B
    
    style D1 fill:#f9f
    style D2 fill:#f9f
    style D3 fill:#f9f
```

### 3.2 FCN（Fully Convolutional Network）—— 语义分割的开端

FCN 将传统 CNN 的全连接层替换为卷积层，实现像素级预测。

```mermaid
flowchart LR
    A[输入图像<br>224×224×3] --> B[Conv1-5<br>传统CNN特征提取]
    B --> C[7×7×4096]
    C --> D[FC6: 1×1 Conv<br>4096→4096]
    D --> E[FC7: 1×1 Conv<br>4096→1000]
    E --> F[反卷积Upconv<br>7×7→224×224, ×32]
    F --> G[Softmax<br>每个像素类别概率]
    G --> H[输出<br>224×224×21]
```

**关键创新**：
- 全连接层改为 1×1 卷积
- 引入转置卷积（反卷积）进行上采样
- Skip Connection 融合浅层细节信息

### 3.3 U-Net（2015）—— 医学图像分割

U-Net 采用了编码器-解码器结构，并添加了跳跃连接来保留空间信息。

```mermaid
flowchart LR
    A[编码器<br>下采样] --> B[解码器<br>上采样]
    
    subgraph encoder_path[编码器路径]
        A
    end
    
    subgraph decoder_path[解码器路径]
        B
    end
    
    A -->|"concat<br>跳跃连接"| B
    
    style A fill:#f9f
    style B fill:#f9f
```

![U-Net架构图](../../assets/figures/fig11-2-unet.png)

**U-Net 特点**：
- 编码器路径捕获上下文信息（全局）
- 解码器路径精确定位（空间）
- 跳跃连接保留细节，防止信息丢失
- 广泛应用于医学影像、卫星图像分割

### 3.4 DeepLab 系列

DeepLab 提出了几个重要技术：

| 技术 | 说明 |
|------|------|
| 空洞卷积（Dilated/Atrous Conv） | 扩大感受野，不损失分辨率 |
| ASPP（Atrous Spatial Pyramid Pooling） | 多尺度空洞卷积捕获多尺度信息 |
| CRF（条件随机场） | 后处理，精细化边界 |

```mermaid
flowchart LR
    A[输入] --> B[Encoder<br>ResNet/CNN + ASPP]
    B --> C[特征图]
    C --> D[Decoder<br>1×1 Conv + 上采样 + 融合]
    D --> E[分割结果]
```

---

## 4. 迁移学习（Transfer Learning）

### 4.1 什么是迁移学习

迁移学习是将在一个任务（源任务）上学到的知识，迁移到另一个相关任务（目标任务）的过程。

**为什么 CNN 迁移学习有效？**

- 浅层卷积核学习的是通用特征（边缘、纹理、颜色）
- 这些特征对各种图像任务都有用
- 不需要从零开始训练整个网络

```mermaid
flowchart LR
    A[ImageNet预训练<br>1000类，140万图像] --> B[提取特征]
    B --> C[学习到的卷积核权重<br>通用视觉特征]
    C --> D[目标任务<br>医学影像分类、宠物识别等]
    D --> E[训练新分类头<br>FC/MLP]
```

### 4.2 迁移学习的策略

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| **特征提取（Feature Extraction）** | 冻结预训练 backbone，只训练新分类头 | 数据量少、类别相似 |
| **微调（Fine-tuning）** | 解冻部分层，较小学习率训练 | 数据量中等 |
| **端到端训练（Full Training）** | 从头训练整个网络（随机初始化） | 数据量大 |

```mermaid
flowchart LR
    A[预训练模型<br>如 ResNet50] --> B[冻结所有卷积层]
    B --> C[特征提取器<br>输出 2048 维特征]
    C --> D[新分类头<br>FC + Softmax]
    D --> E[新类别]
    
    style E fill:#90EE90
```

```mermaid
flowchart LR
    A[预训练模型<br>如 ResNet50] --> B[所有层解冻]
    B --> C[较小学习率<br>全网络训练]
    C --> D[分类头<br>随机初始化训练]
```

### 4.3 迁移学习注意事项

| 注意事项 | 说明 |
|----------|------|
| 数据集相似度 | 源任务与目标任务越相似，迁移效果越好 |
| 学习率调整 | 微调时用较小的学习率（预训练部分的 1/10） |
| 冻结策略 | 数据少时冻结更多层，数据多时可全微调 |
| 预处理一致 | 输入图像的预处理要与预训练时一致 |

---

## 5. 常用预训练模型库

### 5.1 torchvision（PyTorch 官方）

```python
import torchvision.models as models

# 加载预训练模型
resnet = models.resnet50(weights='IMAGENET1K_V1')
vgg = models.vgg16(weights='IMAGENET1K_V1')

# 特征提取（冻结参数）
for param in resnet.parameters():
    param.requires_grad = False

# 修改分类头
resnet.fc = nn.Linear(2048, your_num_classes)
```

### 5.2 timm（PyTorch Image Models）

```python
import timm

# 列表可用模型
models = timm.list_models(pretrained=True)

# 加载模型
model = timm.create_model('efficientnet_b3', pretrained=True)
# 获取特征
model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0)
```

### 5.3 常用模型对比

| 模型 | 参数量 | Top-1 准确率 | 推理速度 |
|------|--------|--------------|----------|
| ResNet-50 | 25M | 76.1% | 快 |
| EfficientNet-B3 | 12M | 81.6% | 中等 |
| ConvNeXt-Small | 50M | 83.1% | 中等 |
| ViT-B/16 | 86M | 79.7% | 较快（Transformer） |
| Swin-T | 28M | 81.3% | 较快 |

---

## 6. CNN 应用总结

```mermaid
flowchart TB
    subgraph cnn_apps[CNN主要应用方向]
        A[图像分类<br>Image Classification] --> A1[输入: 图像 → 输出: 类别概率<br>代表: ResNet、EfficientNet、ConvNeXt]
        
        B[目标检测<br>Object Detection] --> B1[输入: 图像 → 输出: 边界框 + 类别<br>两阶段: Faster R-CNN  单阶段: YOLO、SSD]
        
        C[语义分割<br>Semantic Segmentation] --> C1[输入: 图像 → 输出: 像素级类别标签<br>代表: FCN、U-Net、DeepLab]
        
        D[实例分割<br>Instance Segmentation] --> D1[输入: 图像 → 输出: 像素级类别 + 区分不同实例<br>代表: Mask R-CNN]
        
        E[迁移学习<br>Transfer Learning] --> E1[用预训练模型 + 微调 → 快速适配新任务<br>库: torchvision、timm、TensorFlow Hub]
    end
```

### CNN 应用的关键能力

| 能力 | 说明 |
|------|------|
| 特征提取 | 从底层纹理到高层语义的层次化表达 |
|| 空间感知 | 保留图像的空间结构信息 |
|| 权值共享 | 同一个卷积核检测不同位置的同类特征 |
|| 平移不变 | 对物体位置变化不敏感 |

---

## 本章小结

1. **图像分类**：CNN 的核心应用之一。ImageNet 竞赛推动 CNN 发展——2012 年 AlexNet 首次参赛，2014 年 CNN 超越人类（错误率 5.1%）。常用评价指标：Top-1/Top-5 准确率、参数量、FLOPs、FPS。

2. **目标检测**：识别物体类别并定位（边界框）。两阶段检测器（R-CNN 系列）精度高但慢；单阶段检测器（YOLO 系列）速度快但精度稍低。YOLO 将检测视为单次回归问题，直接预测边界框和类别概率。

3. **语义分割**：像素级分类。FCN 将全连接层改为卷积层引入反卷积上采样；U-Net 采用编码器-解码器结构+跳跃连接保留空间信息；DeepLab 引入空洞卷积扩大感受野。

4. **迁移学习**：用预训练模型（ImageNet 预训练的 ResNet、VGG 等）作为特征提取器或微调，可大幅减少目标任务所需的训练数据和计算资源。策略：特征提取（冻结 backbone）、微调（解冻部分层）、端到端训练。

---

## 知识点思维导图

```mermaid
mindmap
  root((CNN应用))
    图像分类
      ImageNet竟赛推动
      代表: ResNet/EfficientNet/ViT
      指标: Top-1/Top-5准确率
    目标检测
      两阶段 R-CNN系列
        精度高 速度慢
      单阶段 YOLO系列
        速度快 精度接近
      输出: 边界框+类别+置信度
    语义分割
      FCN
        全连接转卷积
        反卷积上采样
      U-Net
        编码器-解码器
        跳跃连接
      DeepLab
        空洞卷积
        ASPP多尺度
    迁移学习
      预训练模型微调
      策略: 特征提取/微调/端到端
      库: torchvision/timm
```

---

## 参考资料

1. Russakovsky, O. et al. (2015). *ImageNet Large Scale Visual Recognition Challenge*
2. Girshick, R. et al. (2014). *Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation* (R-CNN)
3. Ren, S. et al. (2017). *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks* (Faster R-CNN)
4. Redmon, J. et al. (2016). *You Only Look Once: Unified, Real-Time Object Detection* (YOLOv1)
5. Long, J. et al. (2015). *Fully Convolutional Networks for Semantic Segmentation* (FCN)
6. Ronneberger, O. et al. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*
7. He, K. et al. (2017). *Mask R-CNN*
8. Chen, L.C. et al. (2018). *Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation* (DeepLabV3+)