# 从深度学习到 AI 实战：全体系人工智能教程



---

## 📚 教程简介

这是一套面向零基础到中高级学习者的**人工智能完整体系教程**，涵盖从数学基础、机器学习、深度学习，到大语言模型、多模态学习、AI Agent 等各个方向。

**特点：**
- 🐣 从浅入深：无需深厚背景知识，循序渐进
- 📖 图文并茂：善用 mermaid 图表、架构图、流程图
- 🔬 专业准确：每个概念均有公开资料支撑，附参考资料
- 💡 直白易懂：用通俗语言讲解专业概念

**适合人群：**
- AI 初学者，想系统学习人工智能
- 已有一定基础，想深入某个专项方向
- 开发者，想了解 AI 前沿进展
- 产品经理、研究者，想了解 AI 技术原理

---

## 📂 教程结构

```
📦 AI-Tutorial
├── README.md                      ← 你在这里
├── SUMMARY.md                     ← 完整目录（用于文档导航）
├── 附录A-经典模型选型速查表.md    ← 横向对比速查
└── chapters/                      ← 正文章节
    ├── PART0-人工智能导论/
    ├── PART1-数学基础/
    ├── PART2-机器学习基础/
    ├── PART3-深度学习核心/
    ├── PART4-卷积神经网络CNN/
    ├── PART5-序列建模与循环网络/
    ├── PART6-Transformer与预训练/
    ├── PART7-大语言模型LLM/
    ├── PART8-生成式AI/
    ├── PART9-高级学习范式/
    ├── PART10-训练优化与部署/
    ├── PART11-计算机视觉进阶/
    ├── PART12-多模态与Agent/
    ├── PART13-行业应用与前沿/
    ├── PART14-AI伦理与安全/
    ├── PART15-论文阅读与实验写作/
    └── PART16-评测监控与治理/
```

---

## 🗂️ 完整章节导航

### 🌱 入门篇（打牢基础）

| 部分 | 章节 | 内容 |
|------|------|------|
| PART0 | [0-1 人工智能概述](chapters/PART0-人工智能导论/0-1-人工智能概述.md) | 什么是AI？弱/强/超AI分类、图灵测试 |
| PART0 | [0-2 人工智能发展史](chapters/PART0-人工智能导论/0-2-人工智能发展史.md) | 1950s感知机 → 深度学习崛起 → LLM时代（含 ML vs DL 对比、标准化） |
| PART0 | [0-3 AI 学习与项目阶段指南](chapters/PART0-人工智能导论/0-3-AI学习与项目阶段指南.md) | 需求分析、环境分类、骨干选型、训练调优、部署监控 + **训练数据量快速决策**（5.3 节）⭐新增 |
| PART1 | [1-1 线性代数基础](chapters/PART1-数学基础/1-1-线性代数基础.md) | 向量、矩阵、特征值、SVD、张量 |
| PART1 | [1-2 概率论基础](chapters/PART1-数学基础/1-2-概率论基础.md) | 概率分布、贝叶斯定理、MLE |
| PART1 | [1-3 微积分基础](chapters/PART1-数学基础/1-3-微积分基础.md) | 导数、链式法则、梯度下降 |
| PART1 | [1-4 信息论基础](chapters/PART1-数学基础/1-4-信息论基础.md) | 熵、交叉熵、KL散度 |
| PART2 | [2-1 机器学习概述](chapters/PART2-机器学习基础/2-1-机器学习概述.md) | 监督/无监督/半监督/弱监督/自监督 + **聚类分析**（K-Means/DBSCAN/层次）⭐新增 |
| PART2 | [2-2 经典机器学习算法](chapters/PART2-机器学习基础/2-2-经典机器学习算法.md) | 线性回归、决策树、SVM、朴素贝叶斯、KNN |
| PART2 | [2-3 模型评估与验证](chapters/PART2-机器学习基础/2-3-模型评估与验证.md) | 交叉验证、混淆矩阵、ROC/AUC、过拟合/欠拟合 + **混淆矩阵驱动的数据清洗实战**（3.8 节）+ **Epoch/Batch/Iteration 概念 + 学习曲线诊断 + 欠拟合 7 种方案**（3.9-3.11）⭐⭐新增 |
| PART2 | [2-4 特征工程](chapters/PART2-机器学习基础/2-4-特征工程.md) | 特征提取/选择/降维、PCA、标准化（含 VOC/COCO、Albumentations、K 折） |
| PART2 | [2-5 推荐系统与协同过滤](chapters/PART2-机器学习基础/2-5-推荐系统与协同过滤.md) | 协同过滤、矩阵分解、Embedding、NCF、双塔模型、冷启动 ⭐新增 |

### 🔥 核心篇（深度学习）

| 部分 | 章节 | 内容 |
|------|------|------|
| PART3 | [3-1 神经网络基础](chapters/PART3-深度学习核心/3-1-神经网络基础.md) | 感知机、MLP、权重偏差 |
| PART3 | [3-2 激活函数](chapters/PART3-深度学习核心/3-2-激活函数.md) | Sigmoid、ReLU、Tanh、GELU |
| PART3 | [3-3 损失函数](chapters/PART3-深度学习核心/3-3-损失函数.md) | MSE、交叉熵、对比损失 + **交叉熵直观解释 + 信息论视角 + Label Smoothing + Focal Loss**（3.6 节）⭐新增 |
| PART3 | [3-4 正向传播与反向传播](chapters/PART3-深度学习核心/3-4-正向传播与反向传播.md) | 计算图、链式法则、梯度消失/爆炸 |
| PART3 | [3-5 优化算法](chapters/PART3-深度学习核心/3-5-优化算法.md) | GD/SGD、Momentum、Adam、AdamW、学习率策略 + **12 种 SGD 优化器变体大全**（5.5 节）⭐新增 |
| PART3 | [3-6 正则化](chapters/PART3-深度学习核心/3-6-正则化.md) | 过拟合/欠拟合、L1/L2、Dropout、Early Stopping |
| PART3 | [3-7 自编码器](chapters/PART3-深度学习核心/3-7-自编码器.md) | AE/VAE/卷积AE/去噪AE、重建损失、与PCA和扩散模型的关系 ⭐新增 |
| PART4 | [4-1 CNN原理](chapters/PART4-卷积神经网络CNN/4-1-CNN原理.md) | 卷积核、步幅、填充、感受野 + **卷积层如何提取特征 + RNN/CNN/Transformer 三大特征提取器对比** ⭐⭐新增 |
| PART4 | [4-2 CNN核心操作](chapters/PART4-卷积神经网络CNN/4-2-CNN核心操作.md) | 池化、BN、5种归一化（BN/LN/IN/GN/SN）、全连接层 |
| PART4 | [4-3 经典CNN架构](chapters/PART4-卷积神经网络CNN/4-3-经典CNN架构.md) | LeNet→AlexNet→VGG→ResNet→MobileNet→DenseNet→RepVGG + **ResNet50 vs MobileNetV2 vs Inception-ResNet-V2 三大对比** ⭐新增 |
| PART4 | [4-4 CNN 应用全景](chapters/PART4-卷积神经网络CNN/4-4-CNN应用.md) | 四大任务总览 + 分类 + 检测 + 语义分割 + 实例分割 + 骨干映射 + **FPN 深度专题** + FPN 家族演进 + 训练技巧 + 数据集 + 损失/指标（1200+ 行）⭐重写 |
| PART4 | [4-5 现代 CNN 骨干综述](chapters/PART4-卷积神经网络CNN/4-5-现代CNN骨干综述.md) | ResNet-D/RegNet/RepVGG/EfficientNet/ConvNeXt ⭐新增 |
| PART4 | [4-6 注意力机制全景](chapters/PART4-卷积神经网络CNN/4-6-注意力机制全景.md) | SE/ECA/CBAM/CA/SimAM/SK 全覆盖 ⭐新增 |
| PART5 | [5-1 RNN原理](chapters/PART5-序列建模与循环网络/5-1-RNN原理.md) | 循环神经元、BPTT、梯度消失 |
| PART5 | [5-2 RNN变体](chapters/PART5-序列建模与循环网络/5-2-RNN变体.md) | LSTM、GRU、BiLSTM |
| PART5 | [5-3 Seq2Seq模型](chapters/PART5-序列建模与循环网络/5-3-Seq2Seq模型.md) | 编码器-解码器、注意力机制 |
| PART6 | [6-1 注意力机制](chapters/PART6-Transformer与预训练/6-1-注意力机制.md) | Self-Attention、Q/K/V、多头注意力 |
| PART6 | [6-2 Transformer架构](chapters/PART6-Transformer与预训练/6-2-Transformer架构.md) | 编码器-解码器、位置编码 |
| PART6 | [6-3 BERT与预训练语言模型](chapters/PART6-Transformer与预训练/6-3-BERT与预训练语言模型.md) | MLM、NSP、预训练+微调 |
| PART6 | [6-4 GPT系列与生成式预训练](chapters/PART6-Transformer与预训练/6-4-GPT系列与生成式预训练.md) | GPT-1→GPT-4、零样本学习 |
| PART6 | [6-5 注意力机制在CV中的应用 ⭐补充](chapters/PART6-Transformer与预训练/6-5-注意力机制在CV中的应用（补充阅读）.md) | SE/ECA/CBAM/SimAM/SK-Net/Non-Local |
| PART6 | [6-6 词向量与预训练技术 ⭐补充](chapters/PART6-Transformer与预训练/6-6-词向量与预训练技术（补充阅读）.md) | Word2Vec、CBOW/Skip-Gram、MoCo、SimCLR |
| PART6 | [6-7 Transformer变体与高效注意力 ⭐补充](chapters/PART6-Transformer与预训练/6-7-Transformer变体与高效注意力（补充阅读）.md) | Longformer、BigBird、Reformer、Performer、Linformer |
| PART6 | [6-8 CNN在NLP中的应用与膨胀卷积 ⭐补充](chapters/PART6-Transformer与预训练/6-8-CNN在NLP中的应用与膨胀卷积（补充阅读）.md) | TextCNN、膨胀卷积、GLU、ConvNeXt、CoAtNet |
| PART6 | [6-9 混合专家模型MoE与Switch Transformer ⭐补充](chapters/PART6-Transformer与预训练/6-9-混合专家模型MoE与Switch-Transformer（补充阅读）.md) | MoE、Switch Transformer、负载均衡、Mixtral、DeepSeek |

### 🚀 进阶篇（主流技术）

| 部分 | 章节 | 内容 |
|------|------|------|
| PART7 | [7-1 LLM基础与主流模型](chapters/PART7-大语言模型LLM/7-1-LLM基础与主流模型.md) | GPT-4/Claude/LLaMA/Qwen/DeepSeek |
| PART7 | [7-2 LLM训练技术](chapters/PART7-大语言模型LLM/7-2-LLM训练技术.md) | 预训练、SFT、RLHF、DPO + **8 种 PEFT Finetune 方法大全**（LoRA/QLoRA/Adapter/BitFit）⭐新增 |
| PART7 | [7-3 Prompt Engineering](chapters/PART7-大语言模型LLM/7-3-Prompt-Engineering.md) | Zero-shot/Few-shot/CoT/结构化输出/52套模板/去AI味 |
| PART7 | [7-4 LLM进阶技术](chapters/PART7-大语言模型LLM/7-4-LLM进阶技术.md) | LoRA、QLoRA、Flash Attention、RAG |
| PART7 | [7-5 本地大模型与 Ollama 实战](chapters/PART7-大语言模型LLM/7-5-本地大模型与Ollama实战.md) | Ollama 安装/部署/Modelfile/Wechaty/RAG ⭐新增 |
| PART7 | [7-6 大模型厂商全景](chapters/PART7-大语言模型LLM/7-6-大模型厂商全景.md) | 全球四阵营、国内外厂商谱系、开源/闭源商业逻辑、读模型名（参数/MoE）、上下文竞赛、技术路线 ⭐新增 |
| PART7 | [7-7 模型智力与价格性价比](chapters/PART7-大语言模型LLM/7-7-模型智力与价格性价比.md) | 性价比指数、95 模型性价比榜、代际倒挂、开源占高地、月账单视角、帕累托前沿、按任务选型 ⭐新增 |
| PART7 | [7-8 API 按量与订阅制](chapters/PART7-大语言模型LLM/7-8-API按量与订阅制.md) | 按量 vs 订阅、配额窗口（5h/周/月）、价值倍数、隐藏成本、决策树、五个省钱手段 ⭐新增 |
| PART8 | [8-1 生成对抗网络GAN](chapters/PART8-生成式AI/8-1-生成对抗网络GAN.md) | GAN原理、DCGAN、WGAN、StyleGAN |
| PART8 | [8-2 扩散模型Diffusion](chapters/PART8-生成式AI/8-2-扩散模型Diffusion.md) | DDPM、DDIM、CFG、vs GAN对比 |
| PART8 | [8-3 文生图与视频生成](chapters/PART8-生成式AI/8-3-文生图与视频生成.md) | Stable Diffusion、Sora、Runway |
| PART9 | [9-1 小样本学习与元学习](chapters/PART9-高级学习范式/9-1-小样本学习与元学习.md) | Few-shot、MAML、Prototypical Networks |
| PART9 | [9-2 零样本学习与CLIP](chapters/PART9-高级学习范式/9-2-零样本学习与CLIP.md) | CLIP、Zero-shot、Prompt Tuning |
| PART9 | [9-3 强化学习基础](chapters/PART9-高级学习范式/9-3-强化学习基础.md) | MDP、Q-Learning、DQN、PPO、Stable-Baselines3 实战 |
| PART9 | [9-4 图神经网络与知识图谱](chapters/PART9-高级学习范式/9-4-图神经网络与知识图谱.md) | GCN/GraphSAGE/GAT/TransE-TransH-TransR-RotatE/PinSage |
| PART9 | [9-5 联邦学习与隐私保护 ⭐补充](chapters/PART9-高级学习范式/9-5-联邦学习与隐私保护.md) | 联邦学习、隐私计算、FedAvg |
| PART9 | [9-6 知识图谱与智能问答系统 ⭐补充](chapters/PART9-高级学习范式/9-6-知识图谱与智能问答系统.md) | KGQA、SPARQL、多轮问答 |
| PART9 | [9-7 智能问答系统完整 Pipeline](chapters/PART9-高级学习范式/9-7-智能问答系统完整Pipeline.md) | 问句预处理/检索/MRC/生成 ⭐新增 |

### ⚙️ 实战篇（工程部署）

| 部分 | 章节 | 内容 |
|------|------|------|
| PART10 | [10-1 学习率与Batch-Size](chapters/PART10-训练优化与部署/10-1-学习率与Batch-Size.md) | Warmup、Cosine衰减、线性缩放 + **Large Batch 与 LR 关系深度剖析 + 九大学习率衰减策略**（7-8 节）⭐新增 |
| PART10 | [10-2 混合精度与分布式训练](chapters/PART10-训练优化与部署/10-2-混合精度与分布式训练.md) | FP16/BF16、DDP、FSDP、DeepSpeed |
| PART10 | [10-3 模型压缩与优化](chapters/PART10-训练优化与部署/10-3-模型压缩与优化.md) | 知识蒸馏、剪枝、量化、TensorRT |
| PART10 | [10-4 模型部署实战](chapters/PART10-训练优化与部署/10-4-模型部署实战.md) | ONNX/TorchScript/Docker/OpenVINO/TFLite/TensorRT/ONNX Runtime |
| PART10 | [10-5 梯度累加与显存优化 ⭐补充](chapters/PART10-训练优化与部署/10-5-梯度累加与显存优化.md) | 梯度累加 + TF32/BF16/INT8 实测 |
| PART10 | [10-6 OpenMMLab 生态与 MMDeploy 部署](chapters/PART10-训练优化与部署/10-6-OpenMMLab生态与MMDeploy部署.md) | MMDet/MMSeg/MMDeploy 多后端部署 ⭐新增 |
| PART10 | [10-7 边缘部署与 Jetson 实战](chapters/PART10-训练优化与部署/10-7-边缘部署与Jetson实战.md) | Jetson 谱系/TensorRT/性能调优 ⭐新增 |
| PART10 | [10-8 训练硬件选型指南](chapters/PART10-训练优化与部署/10-8-训练硬件选型指南.md) | 个人开发机/CNN 工作站/4-GPU 节点/数据中心/**LLM 个人训练不了** + 云租+二手 + 选型决策树 + 2024-2026 推荐配置 + **算力单位 (TOPS/FLOPS/FLOPs) + 算力 vs 显存非线性 + A100 Tensor Core 黑魔法 + CUDA Compute Capability 80+ GPU 全表** ⭐⭐新增 |
| PART10 | [10-9 训练实战技巧大全](chapters/PART10-训练优化与部署/10-9-训练实战技巧大全.md) | **Google 43 条经验法则** + **Kaggle 39 场实战技巧** + SGD 11 种优化器变体 + Finetune 7 种 PEFT + Snapshot Ensembles + Softmax 温度 T + 训练 Tricks 精华 + 10 步 Checklist ⭐新增 |

### 🎯 应用篇（行业落地）

| 部分 | 章节 | 内容 |
|------|------|------|
| PART11 | [11-1 目标检测进阶](chapters/PART11-计算机视觉进阶/11-1-目标检测进阶.md) | R-CNN→YOLO全系列→Anchor-Free→IoU系列损失→遮挡 + **数据标注原则 1.3 节** + **数据增强详解（Mosaic 4 图拼接）1.4 节** ⭐⭐新增 |
| PART11 | [11-2 语义分割与实例分割](chapters/PART11-计算机视觉进阶/11-2-语义分割与实例分割.md) | FCN/U-Net/DeepLab/Mask R-CNN/SegNet/GCN/BCNet/Mask2Former |
| PART11 | [11-3 图像生成与编辑](chapters/PART11-计算机视觉进阶/11-3-图像生成与编辑.md) | ESRGAN/CycleGAN + CAM/Grad-CAM 可解释性 |
| PART11 | [11-4 神经渲染与3D视觉](chapters/PART11-计算机视觉进阶/11-4-神经渲染与3D视觉.md) | NeRF、3D Gaussian Splatting、SLAM |
| PART11 | [11-5 人脸识别](chapters/PART11-计算机视觉进阶/11-5-人脸识别.md) | 应用场景、数据特点、常见网络、损失函数 |
| PART11 | [11-6 视觉应用补充阅读 ⭐补充](chapters/PART11-计算机视觉进阶/11-6-视觉应用补充阅读.md) | OCR/细粒度/运动检测/行人重识别 ReID |
| PART11 | [11-7 目标跟踪与视频理解](chapters/PART11-计算机视觉进阶/11-7-目标跟踪与视频理解.md) | SORT/DeepSORT/ByteTrack + SlowFast/TimeSformer ⭐新增 |
| PART11 | [11-8 异常检测与工业缺陷](chapters/PART11-计算机视觉进阶/11-8-异常检测与工业缺陷.md) | PatchCore/PaDiM/DRAEM/SimpleNet/MVTec ⭐新增 |
| PART12 | [12-1 多模态学习](chapters/PART12-多模态与Agent/12-1-多模态学习.md) | BLIP、LLaVA、GPT-4V、Gemini |
| PART12 | [12-2 AI Agent 基础](chapters/PART12-多模态与Agent/12-2-AI-Agent基础.md) | 自主规划、Function Calling、记忆系统 |
| PART12 | [12-3 Multi-Agent与具身智能](chapters/PART12-多模态与Agent/12-3-Multi-Agent与具身智能.md) | 多智能体协作、VLA模型、具身AI |
| PART13 | [13-1 AI for Science](chapters/PART13-行业应用与前沿/13-1-AI-for-Science.md) | AlphaFold 2 + **AlphaFold 3 + Boltz-1 + ESM-3 + AI 制药完整链路**（§5）⭐新增 |
| PART13 | [13-2 自动驾驶AI](chapters/PART13-行业应用与前沿/13-2-自动驾驶AI.md) | 端到端、BEV感知、占用网络 + **World Model + VLM-AD（GAIA-2/DriveVLM/EMMA）**（§5）⭐新增 |
| PART13 | [13-3 地理空间AI](chapters/PART13-行业应用与前沿/13-3-地理空间AI.md) | 遥感影像、GIS+深度学习 + **GeoAI 前沿（SatMAE/Prithvi/GeoChat）+ DOTA 旋转框**（§5）⭐新增 |
| PART13 | [13-4 时间序列AI](chapters/PART13-行业应用与前沿/13-4-时间序列AI.md) | Transformer时序预测、金融量化 + **时序基础模型（Chronos/Mamba/Time-LLM/TimeGPT）**（§5）⭐新增 |
| PART13 | [13-5 World Model与AI Safety](chapters/PART13-行业应用与前沿/13-5-World-Model与AI-Safety.md) | 世界模型、具身智能、Constitutional AI |
| PART13 | [13-6 数字孪生 ⭐补充](chapters/PART13-行业应用与前沿/13-6-数字孪生.md) | 数字孪生、仿真、工业应用 + **LLM/World Model/PINN 3 大新范式 + 10 大工业平台**（§8）⭐新增 |

### 🛡️ 伦理篇（安全责任）

| 部分 | 章节 | 内容 |
|------|------|------|
| PART14 | [14-1 AI伦理与对齐](chapters/PART14-AI伦理与安全/14-1-AI伦理与对齐.md) | AI伦理原则/RLHF/Constitutional AI/偏见案例/标准化白皮书 |
| PART14 | [14-2 AI安全与对抗](chapters/PART14-AI伦理与安全/14-2-AI安全与对抗.md) | FGSM/PGD攻击、数据投毒、差分隐私 |
| PART14 | [14-3 数据伦理](chapters/PART14-AI伦理与安全/14-3-数据伦理.md) | 数据偏见/知情同意/标注工人权益/Deepfake/信息茧房/负责任AI ⭐新增 |
| PART15 | [15-1 论文阅读与实验报告写作](chapters/PART15-论文阅读与实验写作/15-1-论文阅读与实验报告写作.md) | 实验报告结构、论文三遍读法、模板 ⭐新增 |

### 🧪 评测篇（可靠性与治理）

| 部分 | 章节 | 内容 |
|------|------|------|
| PART16 | [16-1 模型评测体系与指标](chapters/PART16-评测监控与治理/16-1-模型评测体系与指标.md) | 离线评测、人工评测、在线评测、基准集设计 |
| PART16 | [16-2 RAG 与 Agent 评测](chapters/PART16-评测监控与治理/16-2-RAG与Agent评测.md) | 检索评测、生成评测、任务成功率、工具调用质量 |
| PART16 | [16-3 线上监控与持续回归](chapters/PART16-评测监控与治理/16-3-线上监控与持续回归.md) | 漂移监测、A/B、Canary、Shadow、告警与回归 |
| PART16 | [16-4 AI 治理与红队测试](chapters/PART16-评测监控与治理/16-4-AI治理与红队测试.md) | NIST AI RMF、OWASP 风险、红队测试、上线治理 |

### 📚 附录

| 章节 | 内容 |
|------|------|
| [附录A 经典模型选型速查表](附录A-经典模型选型速查表.md) | 分类/检测/分割/跟踪/异常检测/部署/Jetson/LLM 横向对比 ⭐新增 |

---

## 📊 教程统计

| 项目 | 数值 |
|------|------|
| 总部分数 | 16 |
| 总章节数 | 89 |
| 补充阅读章节数 | 14 |
| 附录 | 1（模型选型速查表） |
| 总行数 | ~51,200 行 |
| Mermaid 图表 | 406 个 |
| 涵盖领域 | 数学基础 → ML → DL → CV → NLP → LLM → 多模态 → Agent → 行业应用 → AI伦理 → 工程实战 → 实验写作 |
| ML-DL 素材覆盖率 | ~60% (从 ~25% 提升) |
| 前沿技术覆盖 | 2024-2026 主流：GRPO / DPO 变体 / V-JEPA 2 / Cosmos / FA3 / Mixture of Depths / Mamba / DiT / Flow Matching / Mechanistic Interpretability / Sleeper Agents / Computer Use / Deep Research 等 |

---

## 🔗 快速导航

- [数学基础](chapters/PART1-数学基础/) - 线性代数、概率论、微积分、信息论
- [机器学习](chapters/PART2-机器学习基础/) - 经典算法、模型评估、特征工程、推荐系统
- [深度学习](chapters/PART3-深度学习核心/) - 神经网络、激活函数、优化器、自编码器
- [CNN](chapters/PART4-卷积神经网络CNN/) - 卷积操作、经典架构、注意力
- [RNN](chapters/PART5-序列建模与循环网络/) - 序列建模、LSTM/GRU、Seq2Seq
- [Transformer](chapters/PART6-Transformer与预训练/) - 注意力机制、BERT、GPT
- [大语言模型](chapters/PART7-大语言模型LLM/) - LLM训练、Prompt、LoRA、RAG、Ollama、厂商全景、性价比、计费方式
- [生成式AI](chapters/PART8-生成式AI/) - GAN、Diffusion、SD、Sora
- [高级学习范式](chapters/PART9-高级学习范式/) - 小样本学习、强化学习、GNN、KGQA
- [训练优化](chapters/PART10-训练优化与部署/) - 分布式训练、模型压缩、部署、OpenMMLab、Jetson
- [计算机视觉](chapters/PART11-计算机视觉进阶/) - 目标检测/分割/跟踪/异常检测/视觉应用
- [多模态与Agent](chapters/PART12-多模态与Agent/) - 多模态学习、AI Agent
- [行业应用](chapters/PART13-行业应用与前沿/) - AI for Science、自动驾驶、Geo AI
- [AI伦理](chapters/PART14-AI伦理与安全/) - 对齐问题、对抗安全、数据伦理、标准化
- [论文与实验](chapters/PART15-论文阅读与实验写作/) - 论文阅读、实验报告
- [评测与治理](chapters/PART16-评测监控与治理/) - 评测体系、RAG/Agent 评测、监控、红队与治理

---

## 这个教程的来源

这个教程里面的一些内容，是我曾经上网课的一些课件，里面还有很多是我的经验和一些网络资料，素材的积累。

曾经一直说要花点时间整理一下，做一个系统性的，完整的扫盲教程，但是苦于没有足够的精力，以及个人的懒惰（当然也有一部分是因为版权的束缚）。现在有了AI agent，吩咐我的agent，花了几块钱的token，整理了一下。重新编写了所有内容，重绘所有的图表，提取知识点，精炼内容等等。

2026 年 6 月 3 日，根据 ML-DL 参考目录（444 个 .md 文件）进行了全面审计和补全，吸收了 80% 以上未利用素材，包括 OpenMMLab 生态、Jetson 边缘部署、Ollama 本地大模型、ReID 行人重识别、目标跟踪 (SORT/ByteTrack)、异常检测 (PatchCore)、现代 CNN 骨干 (RegNet/RepVGG/ConvNeXt)、注意力全景 (SE/CBAM/CA/SK)、智能问答 Pipeline、Grad-CAM 可解释性、TF32/INT8 实测、AI 标准化白皮书等。

2026 年 6 月 24 日，根据 fast.ai Practical Deep Learning for Coders 课程（25 Lessons + Data Ethics）进行内容审计，补充了推荐系统与协同过滤、自编码器、数据伦理、Mean Shift 聚类、Random Erasing、扩散模型图像编辑（Dreambooth/Textual Inversion/DiffEdit/Imagic）、FID/KID 评估指标、Hugging Face Spaces + Gradio 快速部署等 fast.ai 涉及但教程缺失的核心内容。

2026 年 10 月 25 日，补齐 fast.ai 审计遗留的 10 项差距（ULMFiT、lr_find 学习率查找、混淆矩阵驱动数据清洗、从零实现矩阵乘/反向传播/Autograd、OOB 袋外误差、Learner+Callback 框架、图片爬取建数据集、Karras/EDM 采样、LSUV 初始化、Neural Cellular Automata），并新增 **PART7 三个选型与计费章节**：7-6 大模型厂商全景、7-7 模型智力与价格性价比、7-8 API 按量与订阅制。同时修复了多处章节编号错位与统计口径不一致。

现在，公开发布共享。

---
