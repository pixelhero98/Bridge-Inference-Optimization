# 159M step residual corrective：实现说明

本文以已有 E 训练及 reward 微调的实际公式为主。通用代码的接口差异在末尾简述。

## 1. 模型与 rollout

[ResidualDiT](dit.py) 使用 9 层、hidden 1024、16 heads、MLP 3200；原始状态为 32 × 16 × 16，conditioning 维度为 2304，共 **159,196,832 参数**。采用 adaLN-Zero、固定位置编码、起止时间正弦编码；conditioning 经 LayerNorm 后投影，block 的非仿射 LayerNorm 使用 eps 1e-6。调制输出层和 residual 输出层置零，因此初始 residual 为零。接口允许指定状态和 conditioning 维度。

[corrective_rollout](rollout.py) 输入 base 更新后的下一状态，再加一次状态 residual：

$$
z_{\mathrm{base},i+1}=z_i+\Delta t_i\,v_{\mathrm{base}}(z_i,t_i,c),
\qquad
z_{i+1}=z_{\mathrm{base},i+1}
+r_\phi(z_{\mathrm{base},i+1},t_i,t_{i+1},c).
$$

residual 不再乘 $\Delta t$。base 参数冻结，但保留其对当前状态的输入梯度，整条递归轨迹不 detach。

[直接 student](rollout.py) 改为输入当前状态，执行 $z_{i+1}=z_i+r_\phi(z_i,t_i,t_{i+1},c)$；teacher 只提供训练参考，推理不调用 teacher。零 residual 时 corrective 复现 base，直接 student 为恒等映射。

时间网格由调用者提供。E 实验采用同 prompt、同 draw 初始噪声的 K8 teacher 参考和 K2 corrected rollout；CFG 4.5、flow shift 3.0，K2 网格为 $[1,0.75,0]$。

## 2. Loss 与 kernel

### Reward loss

[paired_advantage](objectives.py) 的方向为“corrected 减 baseline”；正值表示改善。E 微调的 PickScore advantage 为：

$$
A_d=\frac{\mathrm{PickScore}_{\mathrm{corrected},d}/26
-\mathrm{detach}(\mathrm{PickScore}_{\mathrm{K8\ base},d}/26)}
{0.05780823156237602}.
$$

baseline 与尺度均冻结；先对同一 context 的 draws 求平均，再对 contexts 平均反传。[TaskLoss](objectives.py) 保留三种 reward loss 形式：

| Reward loss 形式 | 公式 |
|---|---|
| 纯 softplus | $-\mathrm{mean}_d[\tau\,\mathrm{softplus}(A_d/\tau)]$ |
| softplus + linear advantage | $-\mathrm{mean}_d[(1-\alpha)\tau\,\mathrm{softplus}(A_d/\tau)+\alpha A_d]$ |
| 负 advantage 的 softplus | $\mathrm{mean}_d[\tau\,\mathrm{softplus}(-A_d/\tau)]$ |

**E 及其 reward 微调使用以下 reward loss：softplus 内为 $-A/\tau$，外层为正，$\tau=0.15$，没有 linear advantage 项。**

$$
L_{\mathrm{reward}}=\mathrm{mean}_d\left[
\tau\,\mathrm{softplus}\left(-\frac{A_d}{\tau}\right)\right],\qquad \tau=0.15.
$$

它平滑地惩罚 corrected reward 落后于配对 baseline 的情况。本文统一称为 reward loss；源码中对应的类型名仍为 regret。混合形式要求 $0<\alpha<1$；所有形式要求 $\tau>0$。

### Teacher matching kernel

[Laplace kernel](objectives.py) 使用特征的欧氏距离：

$$
k_h(z,y)=\exp\!\left(-\frac{\|f(z)-f(y)\|_2}{h}\right).
$$

每个 context 有两个 generated draws。对应 kernel loss 包含 generated / generated repulsion 和同 draw 的 K8 teacher attraction：

$$
L_K=
\frac12 k_h(z_1,z_2)
-\frac12\left[k_h(z_1,y_1)+k_h(z_2,y_2)\right].
$$

repulsion 的半系数保留；teacher 特征 detach，generated 特征路径保留梯度。E 使用两个独立 kernel：

| 分量 | 特征 | 权重 | 冻结带宽 |
|---|---|---|---|
| latent | latent 展平 | 0.5 | 180.5031280517578 |
| feature | 冻结 DINOv3 ViT-B/16 的最后层 x_norm_clstoken，L2 归一化 | 0.25 | 从冻结校准结果读取 |

0.5 和 0.25 直接乘各自 kernel loss，不归一化。通用接口另支持 ensemble（所有 generated / reference 配对）和 observed（单个观测 target）attraction；E 使用 paired 模式。

### GT 中间状态吸引

K2 的中间时间为 0.75。每个 draw 使用与该 rollout 相同的初始噪声 $\epsilon$，构造：

$$
z_{\mathrm{GT},0.75}=0.25z_{\mathrm{GT}}+0.75\epsilon.
$$

其中 $z_{\mathrm{GT}}$ 为 clean GT latent。这使 GT target 与 corrected 中间状态处于同一噪声时间，并保持 draw 配对。只加入 latent attraction：

$$
T_{\mathrm{GT}}=\mathrm{mean}_d
k_{h_{\mathrm{latent}}}(z_{\mathrm{corrected},0.75,d},z_{\mathrm{GT},0.75,d}).
$$

这一项不包含 GT repulsion；GT final attraction 仅用于诊断，不进入 loss。[通用接口](training.py) 通过 paired reference_path、state_index=1、repulsion=False 表达该项，GT 加噪 target 由调用者提供。

### 实际总损失与权重

E 训练与 PickScore 微调均使用：

$$
L=\lambda_K(0.5L_{K,\mathrm{latent}}+0.25L_{K,\mathrm{feature}})
+\lambda_R L_{\mathrm{reward}}
-w_{\mathrm{GT}}T_{\mathrm{GT}},\qquad \lambda_K=1.
$$

**reward weight $\lambda_R$ 乘在整个 reward loss 外，不放进 advantage 或 softplus 内。** kernel weight $\lambda_K$ 乘 teacher matching；GT attraction 用独立权重 $w_{\mathrm{GT}}$。各阶段只改变 $\lambda_R$ 与 $w_{\mathrm{GT}}$ 的调度，kernel 分量权重不变。

## 3. E 训练与 reward 微调

| 项目 | E 训练 | Reward 微调 |
|---|---|---|
| 更新数 | 5,120 | 追加 1,024，累计 6,144 |
| 使用轨迹 | 40,960 | 追加 8,192，累计 49,152 |
| reward 引入 | $\lambda_R=0$ | 局部第 1 次，即累计第 5,121 次 |
| reward weight | 0 | $\lambda_R^\ast\min(u_{\mathrm{ft}}/256,1)$ |
| kernel weight | 1；latent / feature 为 0.5 / 0.25 | 保持相同 |
| GT 中间吸引 | 前 1,024 次为 0，随后 4,096 次升至 0.1 | 固定 0.1，不重新 warmup |
| reward loss | softplus 内为 $-A/\tau$，$\tau=0.15$，无 linear 项 | 相同 |
| 模型 / AdamW / RNG | seed 42 初始化 | 继承 E 的完整状态 |

E 的 GT 权重按累计更新 $u$ 为：

$$
w_{\mathrm{GT}}(u)=0.1\min\!\left(1,\max\!\left(0,\frac{u-1024}{4096}\right)\right).
$$

第 1,025 次首次大于零，第 5,120 次达到 0.1。微调使用累计编号 $u=5120+u_{\mathrm{ft}}$，因此不重复 warmup。reward 在微调第 256 次（累计第 5,376 次）达到完整系数，随后固定。

无上限 reward 强度扫描在固定的 E checkpoint 上，以 32 个训练 minibatch 校准：

$$
q=\mathrm{median}_{b=1}^{32}
\frac{\|g_{R,b}\|_2}{\|g_{\mathrm{matching},b}\|_2},
\qquad
\lambda_R^\ast=\frac{r}{q},\qquad r\in\{0.1,0.3,1.0\}.
$$

每个 minibatch 为 4 contexts × 2 draws，共 128 个训练 prompts、256 条轨迹。matching 梯度包含 teacher matching 和 GT 吸引；校准不更新参数。$r$ 是初始梯度强度目标，不是 reward 系数，也不是全程动态约束；不设置系数或 endpoint 上限。实际 $\lambda_R^\ast$ 从 calibration/complete.json 的 weights[arm] 读取，feature 带宽从冻结 bandwidth 校准结果读取，缺失值不补造。

两阶段均逐 draw 执行、逐 context 反传 loss/4，再更新一次 AdamW：lr 1e-4、betas (0.9, 0.999)、eps 1e-8、weight decay 1e-4、梯度裁剪 1。微调继承优化器历史与 RNG，延续训练顺序。

## 4. 通用接口说明

[Trainer](training.py) 中，单 reward 对应上述公式时，**Reward.weight=1，TrainingConfig.task_weight 对应外层 $\lambda_R$**；latent / feature 的 Kernel.weight 分别为 0.5 / 0.25，GT kernel 的最大 weight 为 0.1。Reward.weight 是合并多个 advantage 时的内部系数，不代表实验中的外层 reward weight。

当前 task_weight 为常数，尚无 reward ramp、父 checkpoint 微调迁移或梯度强度校准入口；resume=True 只恢复相同配置。历史 IR / CLIP 微调先分别算 loss 再加权，当前通用多 reward 先合并 advantage 再计算一次 loss，两者不等价。

[训练与推理严格 FP32](precision.py)，关闭 autocast、TF32，固定 math SDPA。冻结 base、teacher、reward 和特征模型参数，同时保留 generated 输入梯度；支持 activation checkpointing 与 CPU activation offload，不截断完整轨迹。
