# 图像生成与恢复方法：Metrics、Reward 设计与 Corrective Field 注意事项

## 技术摘要

- **论文指标不等于训练 reward。** FID、KID、FVD 等依赖样本集合的分布指标适合做最终评估或退化 guardrail，通常不适合作为逐样本、可微的训练 reward。
- **Corrective field 应比较配对轨迹。** Base 与 corrected 分支必须复用同一输入、初始噪声、时间网格、条件和随机变量；否则 reward 差值会混入采样噪声。
- **Reward 只能在训练集上定标。** 不同 reward 分量以及不同步数 `K` 的分布可能完全不同，尺度应按分量、按 `K` 独立估计并冻结，测试集不得参与权重或尺度选择。
- **单一 reward 上升不能证明质量全面提升。** 至少同时报告任务 reward、失真/感知指标、分布指标、定性样例和额外计算开销。

## 方法对比

| 类别 | 方法 / Backbone | 方法类型 / Venue | 官方权重 | 推荐 benchmark | 论文使用的 metrics | Reward 设计推荐 metrics |
| --- | --- | --- | --- | --- | --- | --- |
| Image Restoration | [I2SB](https://proceedings.mlr.press/v202/liu23ai.html) | Schrodinger Bridge；ICML 2023 | [官方恢复权重](https://github.com/NVlabs/I2SB) | ImageNet-256 | FID、FID-10k、分类准确率、NFE | 训练：MSE/Charbonnier、LPIPS/DISTS、measurement consistency<br>评估：FID、分类准确率 |
| Image Restoration | [PMRF](https://arxiv.org/abs/2410.00418) | Rectified Flow；ICLR 2025 | [官方 Hugging Face 权重](https://huggingface.co/ohayonguy/PMRF_blind_face_image_restoration) | CelebA-Test、LFW-Test | FID、KID、NIQE、Precision、PSNR、SSIM、LPIPS、Deg、LMD、IndRMSE | 训练：MSE/Charbonnier、LPIPS/DISTS、ArcFace、landmark consistency<br>评估：FID/KID/NIQE/Precision/IndRMSE |
| Video Restoration | [SeedVR](https://arxiv.org/abs/2501.01320) | Diffusion Transformer（Swin-MMDiT）；CVPR 2025 Highlight | [官方 3B/7B 权重](https://huggingface.co/collections/ByteDance-Seed/seedvr) | SPMCS、UDM10、REDS30、YouHQ40、VideoLQ、AIGC38 | PSNR、SSIM、LPIPS、DISTS、NIQE、MUSIQ、CLIP-IQA、DOVER；rFVD（VAE 消融） | 训练：Charbonnier/MSE、LPIPS/DISTS、光流 warping consistency<br>评估：PSNR/SSIM、LPIPS/DISTS、NIQE/MUSIQ/CLIP-IQA、DOVER、rFVD |
| Text to Image | [SANA-0.6B](https://arxiv.org/abs/2410.10629) | Linear DiT / Flow Matching；ICLR 2025 | [官方 600M 512px 权重](https://huggingface.co/Efficient-Large-Model/Sana_600M_512px) | MJHQ-30K、MS-COCO、GenEval、DPG-Bench | FID、CLIP Score、GenEval、DPG-Bench、ImageReward、推理效率 | 训练：按 K 定标的 ImageReward 与 CLIP<br>评估：FID、GenEval、DPG-Bench |
| Image to Image | [UNSB](https://arxiv.org/abs/2305.15086) | Neural Schrodinger Bridge；ICLR 2024 | [官方预训练权重入口](https://github.com/cyclomon/UNSB) | Horse2Zebra、Map2Satellite、Map2Cityscape、Summer2Winter | FID、KID x 100、NFE、推理耗时 | 训练：CLIP 目标域 margin、内容相似度、DINO/LPIPS<br>评估：FID、KID、bootstrap CI、胜率 |

### 指标角色的统一解释

- **逐样本训练 reward**：能够为 corrected 输出提供稳定梯度，例如 MSE、Charbonnier、LPIPS、DISTS、CLIP cosine、ImageReward、ArcFace similarity 和可微的时序一致性。
- **配对监控指标**：对同一输入计算 `metric(corrected) - metric(base)`，例如 PSNR gain、LPIPS reduction、reward advantage 和逐样本胜率。
- **集合级评估 guardrail**：FID、KID、FVD、Precision 等需要一组样本才能可靠估计，只用于选定协议下的验证/测试，不进入逐样本训练目标。
- **计算成本**：报告 NFE 时还要单独报告 corrector 的调用次数、延迟、峰值显存和参数量；相同 NFE 不代表 base 与 corrected 的总计算量相同。

## 通用 Corrective Field 实现原则

### 1. 冻结参数，但不要切断状态梯度

主模型权重设置为不可训练，base rollout 可置于 `no_grad` 中；corrected rollout 仍需要通过主模型对当前状态的 Jacobian 反向传播。不能在每一步对 corrected state `detach()`，也不应使用 teacher forcing 替代真实递归展开。训练后应断言：主模型参数无梯度，corrector 参数和 corrected 输入路径有有限、非零梯度。

### 2. Base 与 corrected 必须严格配对

两条轨迹复用相同的输入、prompt/reference、初始 latent、每步噪声、时间网格、CFG 设置和数值精度。随机桥方法还要复用每个 transition 的随机变量。建议保存这些随机量或由固定 seed 和 sample id 确定性生成，以支持逐样本复算。

### 3. Reward 按分量、按 K 在训练集定标

对 reward 分量 `j` 和步数 `K`，先在 frozen base 的训练样本上估计尺度 `s[j,K]`，再计算：

```text
a[j,K] = (r_corrected[j] - stop_gradient(r_base[j])) / max(s[j,K], eps)
A       = sum_j weight[j] * a[j,K]
loss    = -tau * softplus(A / tau)
```

尺度文件应记录 estimator、样本数和分位数。遇到 floor、饱和或长尾时，不能机械使用 MAD；应检查直方图、floor rate、P1/P99、MAD、标准差及 winsorized 标准差后再选择 estimator。尺度和权重一旦确定，不得根据测试结果调整。

### 4. 用多轴 guardrail 防止 reward hacking

训练过程中至少记录每个 reward 分量的原始值、标准化 advantage、combined advantage、correction norm 和每步状态统计。最终评估同时覆盖：

- 任务目标：ImageReward、CLIP margin、identity similarity 或 reconstruction reward；
- 失真/感知：PSNR、SSIM、LPIPS、DISTS 或 latent MSE；
- 分布质量：FID/KID，视频任务再加 FVD；
- 稳健性：逐样本胜率、配对 bootstrap CI、多 seed 与失败案例；
- 成本：实际延迟、峰值显存、corrector 参数量和额外调用次数。

## I²SB blur-gauss：当前实现与结论

### 实现要点

1. 加载官方 `blur-gauss` EMA 并冻结；完整复用官方 Gaussian blur 退化，不用通用图像库的模糊算子替换。
2. 从同一退化图像独立展开 base 和 corrected 轨迹。该 checkpoint 不使用额外条件输入，因此两条分支都保持 `context=None`。
3. 官方网络输出不是可直接相加的 velocity。当前实验以累计方差定义归一化时间：

   ```text
   S_n = std_fwd[n]^2
   u_n = S_n / S_max
   v_base(x,u) = sqrt(S_max / u) * epsilon_phi(x, t(u))
   v_corrected = v_base + c_theta(x, v_base, u, log2(K))
   ```

   在 `u=1 → u_min` 上使用均匀的 `K` 步 Euler 网格。它与对应端点的确定性 posterior 更新一致，但不应被描述为官方原整数时间网格。
4. Corrector 使用零初始化输出头，并以时间和 `log2(K)` 调制；当前训练联合覆盖 `K=4/8/16`。冻结主网络后使用 activation checkpointing 时，应采用可保留输入梯度的非重入实现并做梯度一致性测试。
5. 当前终点 reward 是把图像映射到 `[0,1]` 后的负 MSE，优化 paired-softplus：

   ```python
   advantage = reward_corrected - reward_base.detach()
   loss = -(0.1 * softplus(advantage / 0.1)).mean()
   ```

### 已核验的先导结果

| 测试范围 | Base PSNR | Corrected PSNR | PSNR gain | 胜率 | 平均 reward gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512 张测试图，K=4/8/16 等权 | 33.187 dB | 35.910 dB | +2.723 dB | 100% | +0.000200 |

分 K 的 PSNR gain 为 `K=4: +2.520 dB`、`K=8: +2.783 dB`、`K=16: +2.865 dB`。训练运行 20 epochs（5,120 次更新），最佳 checkpoint 位于第 20 轮。

### 必须保留的限制

- 这是从 ImageNet validation 重新划分得到的 **单 seed 先导实验**，不是 I²SB 官方数据协议下的复现结果。
- reward gain 的数值很小，`tau=0.1` 下 softplus 在观察区间可能近似线性。必须补充 direct-reward 与 no-K-conditioning 消融，才能分别证明 paired-softplus 和 K conditioning 的贡献。
- 下一阶段除 PSNR/MSE 外，应加入 LPIPS 或 DISTS，并用官方 FID/ResNet-50 accuracy 协议检查是否以牺牲感知或语义质量换取像素指标。

## SANA-0.6B：k=2 长尾决定了定标方式

### 实现要点

1. 冻结官方 SANA-0.6B 512px checkpoint；corrector 对每个 Euler step 的 base velocity 做加性修正，不替换主模型。
2. k=2 与 k=4 使用独立初始化、优化器和 checkpoint，不共享参数或优化状态。两者复用固定的 prompt、初始噪声、时间网格、CFG 和 reward preprocessing。
3. Corrector 输入包含 `[state, base_velocity]`，并以 normalized time、`log2(K)` 和 pooled Gemma caption 调制；输出投影零初始化，以保证训练开始时 corrected 与 base 一致。
4. 当前 reward 为：

   ```text
   a_IR   = (ImageReward_corrected - ImageReward_base) / scale_IR[K]
   a_CLIP = (CLIP_corrected - CLIP_base) / scale_CLIP[K]
   A      = 0.7 * a_IR + 0.3 * a_CLIP
   ```

   然后使用 `tau=0.1` 的 paired terminal softplus。ImageReward 与 OpenCLIP 参数冻结，但 corrected-image 路径必须保留输入梯度；reward encoder 的 resize、antialias 和 normalization 必须固定一致。

### k=2 的 floor 与右长尾

在 1,024 个训练样本的 frozen-base 分布中，k=2 的 ImageReward 有 **79.49%** 落在 `≤ -2.25` 的 floor 附近，但右尾延伸至 `0.723633`：

| 统计量 | k=2 ImageReward base |
| --- | ---: |
| Standard deviation | 0.340800 |
| MAD | 0.005859 |
| Scaled MAD | 0.008687 |
| P1 / P99 | -2.285156 / -0.255901 |
| 1%-99% winsorized std | 0.312051 |
| Floor rate（小于等于 -2.25） | 79.49% |

大量样本堆积在 floor 会让 MAD 几乎塌缩；若用 `0.008687` 做分母，少量右尾样本和微小分数扰动都会产生异常大的标准化 advantage。因此当前 k=2 冻结使用 **1%–99% winsorized std = 0.312051**。k=4 的 floor rate 只有 `6.05%`、分布更宽，使用自己的 scaled MAD `1.386318`；**两者绝不能共用尺度**。

### 当前稳定测试结果

| K | ImageReward（base 到 corrected） | CLIPScore x 100（base 到 corrected） | Combined A | 95% paired CI | A>0 胜率 | Latent MSE 变化 | 500-sample FID（base 到 corrected） |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | -2.199 到 -1.511 | 22.331 到 23.534 | 1.657 | [1.509, 1.808] | 92.0% | 恶化 9.44% | 242.9 到 180.4 |
| 4 | -0.385 到 -0.174 | 28.731 到 28.934 | 0.123 | [0.091, 0.155] | 64.4% | 恶化 6.47% | 129.0 到 120.2 |

### 必须保留的限制

- 两个 K 的组合 reward 都通过门槛，但 latent MSE 同时恶化，说明 reward 改善并不等价于 latent fidelity 改善；必须保留 latent MSE、图像级感知指标和 contact sheet。
- 表中的 FID 仅基于 500 个内部测试样本，不是官方 MJHQ **FID-30K**，不能与论文表格直接比较。
- 不要裁剪运行时 reward 来掩盖长尾，也不要根据测试分布重新选择尺度。应保存原始分量、标准化分量和 floor rate，以便判断改善来自真实排序还是 reward 饱和区的数值效应。
- k=2 与 k=4 的 base 分布差异很大，训练预算、早停和最终结论都应分别给出，不能只报告混合平均值。

## UNSB：保持随机桥配对，并防止过度翻译

### 实现要点

1. 严格复用官方五阶段 stochastic bridge、官方时间表和 `tau=0.01`。每个 `K=1/2/3/4/5` 使用独立 corrector，不共享参数、优化器或 resume state。
2. Base 与 corrected 从同一 source image 开始，并复用每一步 latent/noise；corrected 分支仅在官方 proposal 上加入有界 residual。当前配置的 `max_residual=0.1` 应与 correction RMS、饱和比例一起监控。
3. 当前 OpenCLIP reward 定义为：

   ```text
   r_domain(x)  = cos(x, "a photo of a zebra") - cos(x, "a photo of a horse")
   r_content(x) = cos(x, source_image)

   A = 0.5 * (delta_domain  / scale_domain[K])
     + 0.5 * (delta_content / scale_content[K])
   ```

   每个分量使用该 K 的训练集 base 分布做 1%–99% winsorized 标准差定标，然后使用 `tau=0.1` 的 paired terminal softplus。
4. OpenCLIP 参数、source/base/text embeddings 均冻结并 detach；只有 corrected-output 的 image encoder 路径保留梯度。

### Reward 风险与 guardrail

- **CLIP domain margin 可被投机。** 模型可能通过局部条纹、颜色或纹理提高“zebra-horse”差值，却损坏形状和背景。需要 FID/KID、DINO/LPIPS 或边缘结构一致性以及人工样例共同判断。
- **Content cosine 可能过度保留源域。** 权重过高会留下马的外观，权重过低又会改变姿态、背景和对象布局。应分别报告 domain/content 原始 delta，不能只看 combined advantage。
- **较大 NFE 不一定更好。** UNSB 官方结果已观察到高 NFE 的 over-translation，部分数据集在 NFE 4/5 出现 FID 回升。因此每个 K 必须独立选择，不能默认最大 K 为最终模型。
- **正式验收按 K 独立进行。** 当前协议要求 mean combined advantage `>0`、clustered-bootstrap 95% CI 下界 `>0`、胜率 `>55%`；FID 与 KID×100 只作独立 guardrail。
