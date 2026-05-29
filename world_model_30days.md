

# 30 天世界模型入门打卡清单

> 方向：坑洼/路面场景理解 · 每天约 3-4 小时 · 上午偏读/概念，下午偏动手
> 勾选格式：``- [ ]`` 未完成 / `- [x]` 已完成

---

## 进度总览

| 阶段 | 主题 | 天数 | 状态 |
|------|------|------|------|
| 阶段一 | PyTorch 工程地基 | Day 1-6 |  |
| 阶段二 | 表征学习（世界模型的眼睛） | Day 7-12 | ⬜ |
| 阶段三 | 序列与动态建模（想象力） | Day 13-18 | ⬜ |
| 阶段四 | 世界模型主线（精读+复现） | Day 19-26 | ⬜ |
| 阶段五 | 落地坑洼场景 | Day 27-30 | ⬜ |

---

## 阶段一（Day 1-6）· PyTorch 工程地基

> 目标：让 PyTorch 成为肌肉记忆，把概念翻译成可训练代码。  
> 资源：PyTorch 官方教程 · d2l PyTorch 版 · Karpathy "A Recipe for Training Neural Networks"

### Day 1 · Tensor 与 autograd

- [x] **上午**：学 `torch.Tensor` 创建/索引/广播/形状操作（`view`/`reshape`/`permute`/`unsqueeze`）；理解 `requires_grad`、`grad`、`grad_fn`、`with torch.no_grad()`

- [x] **下午**：纯手写线性回归——不用 `nn.Module`/`optimizer`，构造 `y=3x+2+噪声`，手动 `loss.backward()` 取梯度后手动 `w -= lr*w.grad`，记得 `w.grad.zero_()`

- [x] **产出物**：能收敛、打印 `w≈3, b≈2` 的手写回归代码

  **笔记：**

<!-- 在这里记录当天的sda心得、疑问、发现 -->

---

### Day 2 · nn.Module / optimizer / DataLoader

- [x] **上午**：学 `nn.Module` 的 `__init__` 与 `forward`、`nn.Linear/ReLU/Sequential`、各类 loss、`optim.SGD/Adam`；掌握训练循环五步（前向→算 loss→`zero_grad`→`backward`→`step`）
- [x] **下午**：标准范式重写 Day1 的回归；在 MNIST 上训练一个 3 层 MLP，跑通完整训练/测试循环并打印准确率
- [x] **产出物**：MNIST MLP 测试准确率 >95% 的脚本

**笔记：**

<!-- 在这里记录当天的心得、疑问、发现 -->

---

### Day 3 · CNN 与图像数据管线

- [x] **上午**：理解 `nn.Conv2d/MaxPool2d/BatchNorm2d`、感受野、通道概念；学 `torchvision.transforms`（`ToTensor`、`Normalize`、`RandomCrop`、`RandomHorizontalFlip`）
- [x] **下午**：在 CIFAR-10 上训练一个小 CNN（2-3 个卷积块 + 全连接头），加入数据增强对比"有/无增强"的效果差异
- [x] **产出物**：CIFAR-10 CNN + 有/无增强准确率对比记录

**笔记：**

<!-- 在这里记录当天的心得、疑问、发现 -->

---

### Day 4 · 训练工程化

- [x] **上午**：学 `torch.save/load`、`state_dict`、checkpoint 机制；`.to(device)` 的 GPU 用法；`lr_scheduler`（StepLR/CosineAnnealing）
- [x] **下午**：给 Day3 CNN 加上 checkpoint 保存与断点续训、TensorBoard 记录 loss/acc 曲线、学习率调度
- [x] **产出物**：能中断后从 checkpoint 恢复训练、并在 TensorBoard 看到曲线的版本

**笔记：**

<!-- 在这里记录当天的心得、疑问、发现 -->

---

### Day 5 · 调试与可视化诊断

- [x] **上午**：学过拟合/欠拟合判断（train/val loss 差距）、梯度爆炸/消失排查（打印梯度范数、`clip_grad_norm_`）、常见 bug（忘记 `zero_grad`、`model.train()/eval()` 切换、维度对不齐）
- [x] **下午**：故意制造过拟合（缩小数据集）观察曲线，再用正则/dropout/early stopping 修复；可视化卷积特征图或预测错误样本
- [x] **产出物**：训练问题诊断 checklist 笔记

**笔记：**

<!-- 在这里记录当天的心得、疑问、发现 -->

---

### Day 6 · 阶段复盘与脚手架

- [x] **上午**：回顾前 5 天，整理知识盲点并补齐
- [x] **下午**：把训练逻辑抽象成可复用的 `train.py` 模板（含：参数配置、`build_model`、`build_data`、训练/验证循环、日志、checkpoint）；建好 git 仓库管理整月代码
- [x] **产出物**：干净、带注释的训练脚手架仓库

**笔记：**

<!-- 在这里记录当天的心得、疑问、发现 -->    

---

## 阶段二（Day 7-12）· 表征学习（世界模型的眼睛）

> 目标：吃透 AE → VAE → VQ-VAE，这是后续所有视觉世界模型的编码器基础。  
> 资源：Lilian Weng "From Autoencoder to Beta-VAE" · VQ-VAE 原论文 + 开源实现

### Day 7 · AutoEncoder 原理

- [x] **上午**：学自编码器结构（encoder 压缩 / bottleneck 隐层 / decoder 重建）、重建 loss（MSE/BCE）、欠完备 vs 过完备
- [x] **下午**：在 MNIST 上实现全连接 AutoEncoder，可视化重建结果
- [x] **产出物**：能看到清晰重建数字的 AE

**笔记：**

<!-- 在这里记录当天的心得、疑问、发现 -->

---

### Day 8 · 卷积 AE 与隐空间可视化

- [x] **上午**：学转置卷积 `ConvTranspose2d`/上采样、瓶颈维度对重建质量的影响
- [x] **下午**：实现卷积 AE；把隐变量降到 2 维并画散点图，观察不同数字是否聚类
- [x] **产出物**：卷积 AE + 隐空间散点可视化图

**笔记：**

<!-- 在这里记录当天的心得、疑问、发现 -->

---

### Day 9 · VAE 理论（重点）

- [ ] **上午**：理解 VAE 把隐变量建成概率分布 `q(z|x)`、**重参数化技巧**（`z = μ + σ·ε`，让采样可反传）、ELBO = 重建项 + KL 正则项；推一遍目标函数
  $$
  \mathcal{L}_{VAE} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}\big(q(z|x) \,\|\, p(z)\big)
  $$
  
- [ ] **下午**：读懂一份 VAE 参考实现的每一行（先读后写）

- [ ] **产出物**：能讲清"为什么要重参数化"和"KL 项作用"的手写笔记

**笔记：**

<!-- 在这里记录当天的心得、疑问、发现 -->

---

### Day 10 · VAE 实现与隐空间插值

- [ ] **上午**：实现 VAE（encoder 输出 μ 和 logσ²，重参数化，decoder 重建，loss = 重建 + KL）
- [ ] **下午**：做隐空间插值——取两张图的隐变量做线性插值并逐步解码，观察图像平滑渐变；尝试调 β（KL 权重）看重建/生成的权衡
- [ ] **产出物**：隐空间插值渐变图/动图

**笔记：**

<!-- 在这里记录当天的心得、疑问、发现 -->

---

### Day 11 · VQ-VAE（离散隐变量）

- [ ] **上午**：学 codebook、向量量化（最近邻查表）、straight-through estimator、commitment loss
- [ ] **下午**：读 VQ-VAE 参考实现，理解"图像被编码成一组离散 token"；时间充裕可在 MNIST 上跑通
- [ ] **产出物**：VQ-VAE vs VAE 核心区别与适用场景对比笔记

**笔记：**

<!-- 在这里记录当天的心得、疑问、发现 -->

---

### Day 12 · 阶段复盘：表征对比

- [ ] **上午**：横向对比 AE / VAE / VQ-VAE 的隐空间性质（确定性 vs 概率、连续 vs 离散）与适用模型
- [ ] **下午**：把本阶段 VAE 代码整理为可复用模块（后续坑洼场景实验会直接复用）
- [ ] **产出物**：三者对比表 + 干净的 VAE 可复用模块

| 维度 | AE | VAE | VQ-VAE |
|------|----|-----|--------|
| 隐变量类型 | 确定性 | 连续概率 | 离散 |
| 能否采样生成 | ❌ | ✅ | ✅（配自回归） |
| 隐空间结构 | 无正则 | 平滑/连续 | 离散码本 |
| 典型下游模型 | 分类/检索 | Dreamer/RSSM | IRIS/Transformer系 |

**笔记：**

<!-- 在这里记录当天的心得、疑问、发现 -->

---

## 阶段三（Day 13-18）· 序列与动态建模（想象力）

> 目标：从"看懂一帧"过渡到"预测下一刻"，建模时间动态。  
> 资源：Colah "Understanding LSTM Networks" · "The Illustrated Transformer" · minGPT

### Day 13 · RNN / LSTM / GRU 原理

- [ ] **上午**：学 RNN 隐状态递推、梯度消失问题、LSTM 门控机制（遗忘/输入/输出门）、GRU 简化
- [ ] **下午**：用 `nn.LSTM` 做简单序列任务（正弦波/字符级预测），跑通序列输入输出
- [ ] **产出物**：能预测序列下一个值的 LSTM demo

**笔记：**

<!-- 在这里记录当天的心得、疑问、发现 -->

---

### Day 14 · 序列建模实战与隐状态

- [ ] **上午**：学 `batch_first`、序列 padding、teacher forcing 与自回归推理的区别
- [ ] **下午**：让 Day13 模型做自回归多步预测（用自己的预测作为下一步输入），观察误差累积现象
- [ ] **产出物**：自回归多步预测的曲线（能看到误差随步数累积）

**笔记：**

<!-- 在这里记录当天的心得、疑问、发现 -->

---

### Day 15 · Transformer 与自注意力

- [ ] **上午**：学 self-attention（Q/K/V）、多头注意力、位置编码、为什么能并行且捕捉长依赖
- [ ] **下午**：读懂最小 Transformer 实现（如 minGPT），跑通玩具序列任务
- [ ] **产出物**：能画出/讲清自注意力数据流的笔记

**笔记：**

<!-- 在这里记录当天的心得、疑问、发现 -->

---

### Day 16 · 视频/帧序列与时空建模

- [ ] **上午**：了解 Moving MNIST 数据集、帧序列张量形状 `(B, T, C, H, W)`、ConvLSTM 的思路
- [ ] **下午**：搭好 Moving MNIST 数据管线，可视化输入帧序列
- [ ] **产出物**：能加载并播放 Moving MNIST 序列的数据管线

**笔记：**

<!-- 在这里记录当天的心得、疑问、发现 -->

---

### Day 17 · 未来帧预测（关键练习）

- [ ] **上午**：设计模型：卷积 encoder → LSTM/ConvLSTM → decoder 预测后续帧
- [ ] **下午**：在 Moving MNIST 上训练"用前 N 帧预测后 M 帧"
- [ ] **产出物**：未来帧预测模型 + "预测 vs 真实"对比动画

**笔记：**

<!-- 在这里记录当天的心得、疑问、发现 -->

---

### Day 18 · 阶段复盘：动态建模

- [ ] **上午**：复盘——"VAE 编码 + 序列模型预测隐变量"就是世界模型雏形，把逻辑在脑中串通
- [ ] **下午**：改进 Day17（在隐空间而非像素空间做预测），体会隐空间预测的优势
- [ ] **产出物**："我理解的世界模型最小形态"文字总结（≥200 字）

**笔记：**

<!-- 在这里记录当天的心得、疑问、发现 -->

---

## 阶段四（Day 19-26）· 世界模型主线（精读+复现）

> 读每篇论文都用三问串起来：
> 1. **如何编码观测？** （encoder 结构）
> 2. **如何建模动态？** （dynamics model）
> 3. **如何利用预测能力服务下游任务？** （下游头）

### Day 19 · World Models（Ha & Schmidhuber, 2018）精读

- [ ] **上午**：精读论文，理清 V（VAE 视觉编码）→ M（MDN-RNN 动态预测）→ C（Controller 决策）数据流
- [ ] **下午**：阅读官方/简化复现代码，把每个模块对应到你学过的知识
- [ ] **产出物**：手绘 V-M-C 架构数据流图

**资源：**
- 交互式论文（首选）：https://worldmodels.github.io

**笔记：**

<!-- 在这里记录当天的心得、疑问、发现 -->

---

### Day 20 · World Models 复现/跑通

- [ ] **上午**：配置环境（如 CarRacing），先单独训练 V（VAE）部分
- [ ] **下午**：训练/跑通 M（RNN）部分，理解"在梦境中训练"的含义；C 部分理解原理即可
- [ ] **产出物**：跑通 V 和 M 模块的实验记录

**笔记：**

<!-- 在这里记录当天的心得、疑问、发现 -->

---

### Day 21 · Dreamer 总览与 RSSM

- [ ] **上午**：精读 RSSM（Recurrent State-Space Model）——把隐状态拆成**确定性 `h`** + **随机性 `z`**（32 个 one-hot 向量），理解为什么这么设计
- [ ] **下午**：梳理 Dreamer 整体训练流程：学世界模型 → 在隐空间想象 rollout → 用想象数据训策略
- [ ] **产出物**：RSSM 结构图与文字说明

**资源：**
- DreamerV3 精读解读：https://vitalab.github.io/article/2023/01/19/DreamerV3.html

**笔记：**

<!-- 在这里记录当天的心得、疑问、发现 -->

---

### Day 22 · "在隐空间想象"的思想

- [ ] **上午**：学 Dreamer 如何在 latent space 里做 imagination rollout，actor-critic 如何利用想象轨迹
- [ ] **下午**：研读 DreamerV2/V3 关键改进（离散隐变量、标准化/平衡/变换 trick、稳健性）；理解为何 V3 能"一套超参跨 150+ 任务"
- [ ] **产出物**：Dreamer 数据流全景图（编码→动态→想象→决策）

**资源：**
- Nature 正刊论文（2025）：https://www.nature.com/articles/s41586-025-08744-2
- 项目主页（含方法图与 benchmark）：https://danijar.com/project/dreamerv3

**笔记：**

<!-- 在这里记录当天的心得、疑问、发现 -->

---

### Day 23 · Dreamer 代码走读

- [ ] **上午**：走读官方实现核心模块：world model、RSSM、imagination、actor-critic
- [ ] **下午**：在轻量环境上尝试跑通（跑不动就以读懂数据流为目标）
- [ ] **产出物**：代码模块与论文概念的对应笔记

**资源：**
- 官方仓库（经验证可复现，单卡 A100）：https://github.com/danijar/dreamerv3

**笔记：**

<!-- 在这里记录当天的心得、疑问、发现 -->

---

### Day 24 · Transformer 系世界模型（IRIS / STORM）

- [ ] **上午**：精读 IRIS——VQ-VAE 把帧变 token + Transformer 自回归预测 token 序列；理解范式核心
- [ ] **下午**：结合 STORM 论文中的 Atari 100k 横向对比表，理解 RNN 系 vs Transformer 系的优劣权衡
- [ ] **产出物**：RNN 系 vs Transformer 系世界模型对比笔记

**资源：**
- IRIS 官方仓库（ICLR 2023 notable top 5%）：https://github.com/eloialonso/iris
- STORM 论文（含横向对比表）：https://arxiv.org/pdf/2310.09615

| 维度 | RNN 系（Dreamer/RSSM） | Transformer 系（IRIS/STORM） |
|------|----------------------|--------------------------|
| 长依赖建模 | 一般（隐状态压缩） | 好（直接 attention） |
| 训练速度 | 较快 | 较慢 |
| 可扩展性 | 一般 | 好（scaling law） |
| 隐变量类型 | 连续+离散混合 | 离散 token |

**笔记：**

<!-- 在这里记录当天的心得、疑问、发现 -->

---

### Day 25 · 视觉生成式世界模型（贴近业务）

- [ ] **上午**：了解 Wayve GAIA 系列思路（latent diffusion 世界模型，专为自动驾驶设计，支持多视角一致的可控视频生成）；看 demo 视频建立直觉
- [ ] **下午**：思考"自动驾驶世界模型 = 理解并预测路面与环境演化"与坑洼业务的关联点；记录 3 个可落地的想法
- [ ] **产出物**："生成式世界模型对坑洼场景的启发"要点清单

**资源：**
- GAIA-2 官方介绍：https://wayve.ai/thinking/gaia-2
- 自动驾驶世界模型论文合集（持续更新）：https://github.com/LMD0311/Awesome-World-Model
- 世界模型全景路线图：https://world-model-roadmap.github.io

> ⭐ 额外提示：V-JEPA 2 已有面向汽车安全的真实落地（Nexar BADAS，基于行车记录仪做碰撞预测），这条"表征/预测式世界模型 + dashcam"路线与坑洼检测高度契合，建议顺手了解。

**笔记：**

<!-- 在这里记录当天的心得、疑问、发现 -->

---

### Day 26 · 阶段复盘：世界模型全景

- [ ] **上午**：用"三问"框架把所有读过的模型横向对比，画一张总图
- [ ] **下午**：补齐盲点，整理论文笔记成可检索文档
- [ ] **产出物**：世界模型全景对比图 + 论文笔记库

**世界模型横向对比总表：**

| 模型 | 编码器 | 动态模型 | 隐变量 | 下游任务 | 年份 |
|------|--------|----------|--------|----------|------|
| World Models | VAE | MDN-RNN | 连续 | RL（Evolution Strategies） | 2018 |
| DreamerV1 | CNN | RSSM | 连续 | Actor-Critic | 2020 |
| DreamerV2 | CNN | RSSM | 离散 | Actor-Critic | 2021 |
| DreamerV3 | CNN | RSSM | 离散（one-hot） | Actor-Critic（跨域） | 2023/2025 |
| IRIS | VQ-VAE | Transformer | 离散 token | Actor-Critic | 2023 |
| STORM | VQ-VAE | Transformer | 离散 token | Actor-Critic | 2023 |
| GAIA-1/2 | 视频编码器 | Latent Diffusion | 连续 | 驾驶场景生成 | 2023/2025 |

**笔记：**

<!-- 在这里记录当天的心得、疑问、发现 -->

---

## 阶段五（Day 27-30）· 落地坑洼场景

> 目标：把所学收敛到坑洼/路面场景理解，入职后能快速与团队对话。

### Day 27 · 领域调研：数据集与主流方法

- [ ] **上午**：检索并整理公开数据集（pothole detection / road damage detection），记录标注形式（框/分割/分级）
- [ ] **下午**：梳理主流方法谱系——检测（YOLO 系）、分割（Segmentation 系）、近年多帧/时序方案
- [ ] **产出物**：坑洼场景数据集 + 方法调研文档

**常见公开数据集参考（待核实最新链接）：**

| 数据集 | 类型 | 说明 |
|--------|------|------|
| Pothole-600 | 分类/检测 | 600 张含坑洼路面图像 |
| Road Damage Dataset（RDD） | 检测 | 多国路面损伤，含边界框 |
| CRACK500 | 分割 | 裂缝分割，500 张 |
| DeepCrack | 分割 | 高分辨率裂缝数据 |

**笔记：**

<!-- 在这里记录当天的心得、疑问、发现 -->

---

### Day 28 · 世界模型 × 坑洼的价值假设

- [ ] **上午**：写清三个切入点的论证
  - ① **时空预测**：行驶中预测前方路面状态（传统单帧检测局限 → 世界模型改善）
  - ② **自监督预训练**：用隐空间表征缓解坑洼标注稀缺（标注成本高 → 自监督降低依赖）
  - ③ **生成式仿真**：合成罕见坑洼场景做数据增强（长尾分布 → 生成补充）
- [ ] **下午**：每个切入点写清"传统方法的局限 + 世界模型如何改善 + 验证指标"
- [ ] **产出物**："世界模型为何适合坑洼场景"论证文档（≥500 字）

**笔记：**

<!-- 在这里记录当天的心得、疑问、发现 -->

---

### Day 29 · 最小可行实验（MVP）

- [ ] **上午**：找一个小型路面/坑洼图像数据集，复用阶段二整理的 VAE 模块搭好训练管线
- [ ] **下午**：做无监督路面表征学习，把隐空间可视化，观察正常路面 vs 坑洼在隐空间是否可分
- [ ] **产出物**：路面 VAE 实验 + 隐空间可视化结果（散点图/t-SNE）

**笔记：**

<!-- 在这里记录当天的心得、疑问、发现 -->

---

### Day 30 · 总结与入职规划

- [ ] **上午**：整理整月代码仓库与笔记，输出"我对世界模型 + 坑洼场景的理解"总文档
- [ ] **下午**：列出 3-5 个入职后想和团队验证的技术假设
- [ ] **产出物**：总结文档 + 待验证假设清单

**我的技术假设（待填写）：**

1. 
2. 
3. 
4. 
5. 

**笔记：**

<!-- 在这里记录当天的心得、疑问、发现 -->

---

## 核心资源汇总

### 论文与项目

| 资源 | 链接 | 对应天数 |
|------|------|----------|
| World Models 交互式论文 | https://worldmodels.github.io | Day 19-20 |
| DreamerV3 Nature 正刊 | https://www.nature.com/articles/s41586-025-08744-2 | Day 22 |
| DreamerV3 项目主页 | https://danijar.com/project/dreamerv3 | Day 22 |
| DreamerV3 官方仓库 | https://github.com/danijar/dreamerv3 | Day 23 |
| DreamerV3 精读解读 | https://vitalab.github.io/article/2023/01/19/DreamerV3.html | Day 21 |
| IRIS 官方仓库 | https://github.com/eloialonso/iris | Day 24 |
| STORM 论文 | https://arxiv.org/pdf/2310.09615 | Day 24 |
| GAIA-2 介绍 | https://wayve.ai/thinking/gaia-2 | Day 25 |
| 自动驾驶世界模型合集 | https://github.com/LMD0311/Awesome-World-Model | Day 25 |
| 世界模型全景路线图 | https://world-model-roadmap.github.io | Day 26 |

### 学习资料

| 资源 | 类型 | 对应阶段 |
|------|------|----------|
| PyTorch 官方教程 | 文档 | 阶段一 |
| d2l（Dive into Deep Learning）PyTorch 版 | 书 | 阶段一-二 |
| Karpathy "A Recipe for Training Neural Networks" | 博客 | 阶段一 |
| Lilian Weng "From Autoencoder to Beta-VAE" | 博客 | 阶段二 |
| Colah "Understanding LSTM Networks" | 博客 | 阶段三 |
| "The Illustrated Transformer" | 博客 | 阶段三 |
| minGPT | 代码 | 阶段三 |

---

## 整月产出物清单

| 天数 | 产出物 | 状态 |
|------|--------|------|
| Day 1 | 手写线性回归（w≈3, b≈2 收敛） | ⬜ |
| Day 2 | MNIST MLP >95% | ⬜ |
| Day 3 | CIFAR-10 CNN + 增强对比 | ⬜ |
| Day 4 | 带 checkpoint + TensorBoard 的版本 | ⬜ |
| Day 5 | 训练诊断 checklist | ⬜ |
| Day 6 | 可复用 train.py 脚手架 + git 仓库 | ⬜ |
| Day 7 | MNIST AE 重建 | ⬜ |
| Day 8 | 卷积 AE + 隐空间散点图 | ⬜ |
| Day 9 | 重参数化与 KL 笔记 | ⬜ |
| Day 10 | VAE 隐空间插值图/动图 | ⬜ |
| Day 11 | VQ-VAE vs VAE 对比笔记 | ⬜ |
| Day 12 | 表征对比表 + 可复用 VAE 模块 | ⬜ |
| Day 13 | LSTM 序列预测 demo | ⬜ |
| Day 14 | 自回归多步预测曲线 | ⬜ |
| Day 15 | 自注意力数据流笔记 | ⬜ |
| Day 16 | Moving MNIST 数据管线 | ⬜ |
| Day 17 | 未来帧预测对比动画 | ⬜ |
| Day 18 | 世界模型最小形态总结 | ⬜ |
| Day 19 | V-M-C 数据流图 | ⬜ |
| Day 20 | V+M 模块实验记录 | ⬜ |
| Day 21 | RSSM 结构图与说明 | ⬜ |
| Day 22 | Dreamer 全景数据流图 | ⬜ |
| Day 23 | 代码模块与论文对应笔记 | ⬜ |
| Day 24 | RNN 系 vs Transformer 系对比笔记 | ⬜ |
| Day 25 | 生成式世界模型对坑洼的启发清单 | ⬜ |
| Day 26 | 世界模型全景对比图 + 笔记库 | ⬜ |
| Day 27 | 坑洼数据集 + 方法调研文档 | ⬜ |
| Day 28 | 价值假设论证文档 | ⬜ |
| Day 29 | 路面 VAE 实验 + 隐空间可视化 | ⬜ |
| Day 30 | 总结文档 + 技术假设清单 | ⬜ |

---

*最后更新：2026-05-30 | 在线打卡版：https://go.tabbit.space/projects/1746bb34-3747-4c29-99cf-a927557277bf/index.html*
