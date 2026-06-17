# AlexNet-CIF10-CatDog 项目文档

## 1. 项目简介

### 项目名称
AlexNet-CIF10-CatDog

### 项目定位
基于 PyTorch 从零实现 AlexNet，在 CIFAR-10 数据集上提取猫狗图片进行二分类训练，提供 Tkinter 图形化推理界面。

### 项目目标
- 理解经典 CNN 架构 AlexNet 的结构与原理
- 掌握 PyTorch 训练全流程：数据准备 → 模型搭建 → 训练 → 评估 → 部署
- 实践图像分类的完整 pipeline

### 核心功能
- CIFAR-10 猫狗图片提取与数据集划分
- AlexNet 二分类模型训练（含数据增强、学习率调度）
- Loss/Accuracy 训练曲线自动生成
- Tkinter GUI 交互式推理预测

---

## 2. 技术栈

| 领域 | 技术 |
|------|------|
| 深度学习框架 | PyTorch 2.7+、torchvision |
| 数据处理 | numpy、Pillow |
| 可视化 | matplotlib |
| GUI | tkinter |
| 硬件加速 | CUDA (NVIDIA GPU) |

---

## 3. 项目结构

```text
AlexNet/
├── net.py                  # AlexNet 模型定义 (MyAlexNet)
├── train.py                # 训练脚本
├── test01.py               # Tkinter GUI 推理脚本
├── prepare_data.py         # 从 CIFAR-10 pickle 提取猫狗图片
├── split_data.py           # 8:2 划分训练/验证集
├── acc_curve.png           # 准确率曲线 (自动生成)
├── loss_curve.png          # Loss 曲线 (自动生成)
├── .gitignore
├── README.md
├── docs/
│   └── PROJECT_GUIDE.md    # 本文件
├── save_model/             # 模型权重 (gitignored)
│   ├── best_model1.pth     # 验证集最佳模型
│   └── last_model1.pth     # 最后一轮模型
├── cifar10/                # CIFAR-10 原始数据 (gitignored)
│   └── cifar-10-batches-py/
│       ├── data_batch_1~5
│       ├── test_batch
│       └── batches.meta
├── data_name/              # 提取的猫狗原图 (gitignored)
│   ├── cat/ (500张)
│   └── dog/ (500张)
└── data/                   # 划分后的训练/验证集 (gitignored)
    ├── train/ (cat 400 + dog 400)
    └── val/   (cat 100 + dog 100)
```

---

## 4. 环境要求

### 软件版本

| 组件 | 推荐版本 | 最低版本 |
|------|---------|---------|
| Python | 3.10 | 3.8+ |
| PyTorch | 2.7+ | 1.10 |
| torchvision | 0.22+ | 0.11 |
| CUDA | 12.x / 11.8 | 11.x |
| matplotlib | 3.10 | 3.0 |
| numpy | 2.2 | 1.21 |

### 快速安装

```bash
# 使用 conda
conda create -n ml2026 python=3.10
conda activate ml2026
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib numpy pillow

# 或使用清华镜像加速
pip install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 5. 安装部署

### 克隆项目

```bash
git clone https://github.com/Jared-Linn/AlexNet-cif10-catdog.git
cd AlexNet-cif10-catdog
```

### 获取数据

CIFAR-10 数据集不在仓库中（.gitignore 排除），需要手动获取：

**方式 A：GitCode 镜像（国内推荐）**
```bash
git clone https://gitcode.com/open-source-toolkit/94ecd.git tmp
unrar x tmp/cifar-10-python.tar.gz cifar10/
rm -rf tmp
```

**方式 B：官网下载**
```bash
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzf cifar-10-python.tar.gz -C cifar10/
```

### 数据准备

```bash
python prepare_data.py    # 从 pickle 提取猫狗各 500 张
python split_data.py      # 8:2 划分训练/验证集
```

---

## 6. 模型说明

### 模型结构

MyAlexNet 是经典 AlexNet 的简化单路版，输入 3×224×224，输出 2 类：

| 层 | 类型 | 参数 | 输出尺寸 | 备注 |
|---|------|------|---------|------|
| c1 | Conv2D | 3→48, 11×11, s=4, p=2 | 48×54×54 |
| c2 | Conv2D | 48→128, 5×5, s=1, p=2 | 128×54×54 |
| s2 | MaxPool2D | k=2 | 128×27×27 |
| c3 | Conv2D | 128→192, 3×3, s=1, p=1 | 192×27×27 |
| s3 | MaxPool2D | k=2 | 192×13×13 |
| c4 | Conv2D | 192→192, 3×3, s=1, p=1 | 192×13×13 |
| c5 | Conv2D | 192→128, 3×3, s=1, p=1 | 128×13×13 |
| s5 | MaxPool2D | k=3, s=2 | 128×6×6 |
| f6 | Linear | 4608→2048 | 2048 | + Dropout(0.5) |
| f7 | Linear | 2048→2048 | 2048 | + Dropout(0.5) |
| f8 | Linear | 2048→1000 | 1000 | + Dropout(0.5) |
| f9 | Linear | 1000→2 | 2 | 输出 logits |

### 与原版 AlexNet 差异

- 去掉了双 GPU 分支，改为单路
- 卷积核数量减半（96→48, 256→128, 384→192）
- Conv1 后缺少 MaxPool（原版有，当前版本无）
- 全连接层从 4096→4096→1000 改为 2048→2048→1000→2

### 训练配置

| 参数 | 值 |
|------|-----|
| 输入尺寸 | 3×224×224 (CIFAR 32×32 上采样) |
| Batch size | 16 |
| 优化器 | SGD (lr=0.01, momentum=0.9) |
| 学习率调度 | StepLR, step=10, gamma=0.5 |
| 损失函数 | CrossEntropyLoss |
| Epochs | 100 |
| 数据增强 | RandomHorizontalFlip, ColorJitter, RandomRotation(10°) |

### 训练结果

| 指标 | 值 |
|------|-----|
| 最佳验证准确率 | 75.48% (epoch 52) |
| 最佳验证 Loss | 0.537 |
| 训练 Loss (最终) | ~0.03 |
| 验证 Loss (最终) | ~0.68 |

**已知问题：**
- CIFAR-10 原图 32×32 上采样到 224×224，信息损失大
- 约 epoch 50 后出现过拟合（val_loss 开始震荡上升）
- Conv1 后缺少 MaxPool（与原始 AlexNet 不一致）

---

## 7. 启动项目

### 训练

```bash
python train.py
```

自动保存最佳模型到 `save_model/best_model1.pth`，训练结束后生成 loss/acc 曲线图。

### GUI 推理

```bash
python test01.py
```

功能：
- 选择本地图片 → 预测猫/狗
- 随机测试验证集 10 张（结果打印到控制台）

---

## 8. 功能说明

### 8.1 数据准备 (prepare_data.py)

**输入：** `cifar10/cifar-10-batches-py/data_batch_1~5` (CIFAR-10 pickle 文件)
**处理流程：**
1. 用 pickle 读取 50000 张训练图片
2. 提取 label=3 (cat) 和 label=5 (dog) 的图片
3. 保存为 PNG 到 `data_name/{cat,dog}/`

**输出：** 猫狗各 500 张 32×32 PNG 图片

### 8.2 数据集划分 (split_data.py)

**输入：** `data_name/` 下猫狗各 500 张
**处理流程：** 随机抽取 20% 作为验证集，80% 作为训练集
**输出：** `data/train/{cat,dog}/` 各 400 张，`data/val/{cat,dog}/` 各 100 张

### 8.3 模型训练 (train.py)

**输入：** `data/train/` 和 `data/val/`
**处理流程：**
1. 数据增强：Resize→224×224, RandomHorizontalFlip, ColorJitter, RandomRotation
2. 归一化：Normalize(mean=0.5, std=0.5)
3. SGD 优化，StepLR 学习率衰减
4. 每轮保存最佳模型
**输出：** 模型权重、loss_curve.png、acc_curve.png

### 8.4 GUI 推理 (test01.py)

**输入：** 用户选择的 jpg/png 图片
**处理流程：**
1. 加载预训练模型权重
2. 图片预处理（与训练一致的 transform）
3. 前向推理，取 argmax 得到类别
**输出：** 界面显示 "cat" 或 "dog"

---

## 9. 常见问题

### 9.1 CUDA 不可用

```bash
# 检查 CUDA
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# 安装匹配版本的 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 9.2 CIFAR-10 下载失败

国内网络访问 `cs.toronto.edu` 可能超时。解决方案：
- 使用 GitCode 镜像（见上文"获取数据"）
- 使用 HTTP 协议（禁用 SSL）：`curl --insecure -L -O http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz`

### 9.3 DLL 加载错误 (Windows)

```
ImportError: DLL load failed while importing _lzma
```
- 从其他正常环境复制 `liblzma.dll` 到 `envs/<name>/Library/bin/`

### 9.4 conda SSL 错误

```bash
conda config --set ssl_verify false
# 或使用 pip 替代 conda 安装
```

---

## 10. 开发规范

### Git 提交规范

```text
feat: 新功能
fix: 修复问题
docs: 文档更新
enh: 功能增强/改进
chore: 其他修改
```

### 分支规范

```text
main        — 稳定版本
feature/*   — 新功能开发
fix/*       — 问题修复
```

---

## 11. 发布记录

| 版本 | 日期 | 内容 |
|------|------|------|
| v0.1.0 | 2026-06-16 | 初始提交：AlexNet 模型、训练脚本、GUI |
| v0.2.0 | 2026-06-17 | 离线数据加载、CIFAR-10 pickle 直读 |
| v0.3.0 | 2026-06-18 | 数据增强改进 (HFlip+ColorJitter+Rotation)，val_acc 68%→75% |

---

## 12. Roadmap

### 当前阶段
- [x] AlexNet 模型定义与训练 pipeline
- [x] CIFAR-10 猫狗数据提取与划分
- [x] Tkinter GUI 交互推理
- [x] Loss/Acc 训练曲线
- [x] 数据增强优化

### 下一阶段
- [ ] 修复 Conv1 后缺少 MaxPool 的问题（对齐原始 AlexNet）
- [ ] weight decay 正则化进一步抑制过拟合
- [ ] 尝试直接用 32×32 输入（跳过上采样）
- [ ] 模型评估：混淆矩阵、PR 曲线
- [ ] 模型导出：ONNX 格式

---

## 13. 参考

- [AlexNet 原论文](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)
- [CIFAR-10 数据集](https://www.cs.toronto.edu/~kriz/cifar.html)
- [PyTorch 官方文档](https://pytorch.org/docs/stable/)
- [GitCode CIFAR-10 镜像](https://gitcode.com/open-source-toolkit/94ecd/)

---

文档最后更新时间：2026-06-18
