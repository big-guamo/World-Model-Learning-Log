# World-Model-Learning-Log

---
以下是由 Claude 生成的 30 天世界模型入门学习计划，项目中的 dayX并不严格对于下面的计划。
### **第一阶段（Day 1–6）：PyTorch 工程能力速成**

世界模型的所有组件——VAE、RNN/Transformer、策略网络——都需要用 PyTorch 实现，所以第一周必须把框架打通。这一阶段的目标不是看完所有 API，而是能独立写出「数据加载 → 模型定义 → 训练循环 → 验证 → 保存」的完整流程。

**Day 1：张量与自动求导**
学习 `torch.Tensor` 的创建、形状变换、广播机制，理解 `requires_grad`、`backward()`、`grad` 三者关系。手写一个一元线性回归，用 `loss.backward()` 手动更新参数，不调用 optimizer，体会自动微分的本质。
·
**Day 2：`nn.Module` 与优化器**
掌握 `nn.Linear`、`nn.Sequential`、`forward` 写法，学习 `torch.optim`（SGD、Adam）和常见 loss 函数。把 Day 1 的线性回归改写为标准 `nn.Module` 风格。

**Day 3：Dataset 与 DataLoader**
理解 `Dataset`、`DataLoader`、`transforms` 三件套，学会自定义 Dataset。下载 MNIST，写出完整训练脚本。

**Day 4：MLP 实战 + GPU 训练**
在 MNIST 上训练一个 3 层 MLP，加入 `device = "cuda"`、`model.to(device)`、混合精度（可选 `torch.cuda.amp`），跑通 GPU 训练。

**Day 5：CNN 与卷积**
学习 `Conv2d`、`BatchNorm2d`、`MaxPool2d`，在 CIFAR-10 上训练一个简易 CNN，理解卷积输出尺寸公式 $H_{out} = \lfloor (H_{in} + 2p - k)/s \rfloor + 1$。

**Day 6（周末小项目）：复现一个完整 Pipeline**
用 PyTorch 复现 LeNet 或 ResNet-18 的简化版，加入 `tensorboard` 或 `wandb` 记录 loss 曲线，保存 checkpoint，写出 `train.py / model.py / dataset.py` 的工程化目录结构。

推荐资料：[PyTorch 官方教程](https://pytorch.org/tutorials/)、[d2l.ai](https://d2l.ai/)（中文版叫《动手学深度学习》）。

---

### **第二阶段（Day 7–12）：生成模型基础——VAE**

世界模型 V 模块（Vision）的核心是 **变分自编码器（VAE）**，用来把高维图像压缩成低维潜变量 $z$。这一阶段必须吃透 VAE 的数学和实现。

**Day 7：自编码器（AE）**
先实现一个普通 AE，在 MNIST 上压缩到 32 维 latent，再重建。理解「重建损失」与「瓶颈」。

**Day 8：VAE 数学推导**
推导 ELBO：

$$
\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \,\|\, p(z))
$$

理解「重参数化技巧」 $z = \mu + \sigma \odot \epsilon, \epsilon \sim \mathcal{N}(0, I)$ 为何能让梯度回传。

**Day 9：VAE 实现（MNIST）**
用 PyTorch 实现 encoder 输出 $\mu, \log\sigma^2$、decoder 重建图像，loss = BCE + KL，跑通 MNIST。

**Day 10：卷积 VAE（CIFAR-10 / CarRacing 帧）**
把 encoder/decoder 换成卷积结构，处理 $64 \times 64$ RGB 图像。这是世界模型 V 模块的标准输入尺寸。

**Day 11：β-VAE 与 latent 可视化**
引入 β 系数控制解耦程度，用 t-SNE 或 UMAP 可视化 latent 空间，观察相似图像是否聚类。

**Day 12（周末小项目）：在游戏帧上训练 VAE**
用 `gymnasium` 安装 `CarRacing-v2`，随机策略采集 10000 帧，resize 到 $64 \times 64$，训练 VAE 把图像压成 32 维 $z$。这就是 Ha & Schmidhuber 论文里 V 模块的真实任务。

参考：[Auto-Encoding Variational Bayes (Kingma, 2013)](https://arxiv.org/abs/1312.6114)。

---

### **第三阶段（Day 13–18）：序列建模——RNN / LSTM / MDN-RNN**

世界模型的 M 模块（Memory）用 **MDN-RNN**（混合密度网络 + LSTM）来预测下一帧的 latent 分布。这一阶段重点是序列建模和混合高斯输出。

**Day 13：RNN / LSTM 基础**
理解 RNN 的展开图、梯度消失问题、LSTM 的三个门。在 PyTorch 里用 `nn.LSTM` 做字符级语言模型（如生成莎士比亚文本）。

**Day 14：序列预测任务**
用 LSTM 预测正弦波、股票时间序列等连续值，理解 `batch_first`、`hidden_state` 的传递。

**Day 15：混合密度网络（MDN）**
学习 MDN：网络输出 $K$ 个高斯分量的 $\pi_k, \mu_k, \sigma_k$，loss 是负对数似然：

$$
\mathcal{L} = -\log \sum_{k=1}^{K} \pi_k \, \mathcal{N}(y \mid \mu_k, \sigma_k^2)
$$

在玩具数据（一对多映射，比如 $x = y + 0.3\sin(2\pi y) + \epsilon$）上验证。

**Day 16：MDN-RNN 实现**
把 LSTM 的输出接到 MDN 头，输入是 $(z_t, a_t)$，输出预测 $z_{t+1}$ 的混合高斯分布。先在合成序列上跑通。

**Day 17：在 CarRacing latent 序列上训练 MDN-RNN**
用 Day 12 的 VAE 把采集的帧序列编码成 $z$ 序列，加上动作 $a$，训练 MDN-RNN 学会预测下一时刻的 $z$。这是 World Models 论文 M 模块的核心。

**Day 18（周末小项目）：Dream Rollout**
固定 V 和 M，从某个真实帧出发，让 MDN-RNN 自回归生成 latent 序列，再用 VAE decoder 还原成图像，可视化「梦境」。这是世界模型最有趣的部分。

---

### **第四阶段（Day 19–22）：强化学习与控制器（C 模块）**

世界模型的 C 模块（Controller）通常很小，原论文用 **CMA-ES** 进化策略训练。但现代复现也常用 PPO，所以这两条路都要了解。

**Day 19：RL 基础概念**
快速过 MDP、价值函数、策略梯度、Q-learning。推荐 [Spinning Up in Deep RL](https://spinningup.openai.com/) 的核心概念章节。

**Day 20：用 PPO 玩 CartPole**
用 `stable-baselines3` 或自己写 PPO，在 `CartPole-v1` 上训练，理解 actor-critic 与 GAE。

**Day 21：CMA-ES 进化策略**
学习 CMA-ES 算法思路（不需要手写，用 `cma` 库），理解为什么世界模型用进化算法训练只有几百参数的小型 controller——因为参数少，无梯度搜索反而高效。

**Day 22：Controller 拼装**
定义 controller：输入 $[z_t, h_t]$（VAE latent + LSTM hidden），输出动作 $a_t$。在真实 CarRacing 环境中用 CMA-ES 优化 controller 权重，跑通完整 V+M+C 流水线。

---

### **第五阶段（Day 23–27）：复现 Ha & Schmidhuber《World Models》**

到这一阶段，所有组件都已具备，集中精力把 2018 年原论文跑通。

**Day 23：精读论文**
阅读 [Ha & Schmidhuber, World Models (2018)](https://arxiv.org/abs/1803.10122)，配合作者的[交互式博客](https://worldmodels.github.io/)，理解三个模块如何协作、为什么要在「梦中」训练。

**Day 24：在真实环境训练 controller**
用真实 CarRacing 环境 + V + M 提取的 $[z, h]$ 作为 controller 输入，CMA-ES 优化，目标 reward ≥ 600。

**Day 25：在「梦境」中训练 controller**
完全脱离真实环境，让 controller 在 MDN-RNN 想象的 latent 序列里学习，然后迁移回真实环境测试。这是论文最 striking 的结论。

**Day 26：调参与温度系数 τ**
研究 MDN-RNN 采样温度 τ 对梦境稳定性的影响，理解「想象越随机，policy 越鲁棒」的现象。

**Day 27：写实验报告**
把复现结果整理成 Notebook 或博客，记录每个模块的训练曲线、最终 reward、失败案例。

参考开源实现：[hardmaru/WorldModelsExperiments](https://github.com/hardmaru/WorldModelsExperiments)、[ctallec/world-models](https://github.com/ctallec/world-models)（PyTorch 版，更适合你）。

---

### **第六阶段（Day 28–30）：现代世界模型前沿**

2018 年之后，世界模型快速演进，以下三条主线必须了解，否则知识会停留在 7 年前。

**Day 28：Dreamer 系列（V1/V2/V3）**
阅读 [DreamerV3 (Hafner et al., 2023)](https://arxiv.org/abs/2301.04104)，理解 RSSM（Recurrent State-Space Model）如何用确定性 + 随机 latent 改进 MDN-RNN，以及 actor-critic 如何在 latent 空间里端到端训练（不再用 CMA-ES）。DreamerV3 是目前用一套超参数通杀 150+ 任务的标杆。

**Day 29：Transformer 世界模型**
阅读 [IRIS (Micheli et al., 2023)](https://arxiv.org/abs/2209.00588) 和 [TWM / TransDreamer]，理解为什么用 Transformer 替换 LSTM，以及 token 化图像（VQ-VAE）如何让世界模型变成「序列预测」问题。这一思路与大语言模型范式打通。

**Day 30：通用世界模型与展望**
浏览 [Genie (DeepMind, 2024)](https://sites.google.com/view/genie-2024/)、[Sora 技术报告](https://openai.com/index/video-generation-models-as-world-simulators/)、[V-JEPA / V-JEPA 2 (Meta)](https://ai.meta.com/research/publications/v-jepa-2/) 等近期工作，理解视频生成模型、JEPA 自监督范式与传统 RL 世界模型的异同。最后写一份「我对世界模型的理解」笔记，定下后续 3 个月的深入方向（如机器人控制、视频生成、自动驾驶仿真等）。

---
