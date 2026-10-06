# SeedVR：Softplus 与 Linear 对比

冻结 SeedVR-3B，只训练修正场。当前保留 **Softplus** 和 **Linear** 两种奖励项参数化。

## 1. 公式

令 X₀ 为原模型输出，Xφ 为修正后的输出，y 为目标视频；两条轨迹使用相同输入、初始噪声和 NFE。定义相对原模型的恢复提升：

```math
A=\sum_{j=1}^{3}w_j\frac{\ell_j(X_0,y)-\ell_j(X_\phi,y)}{s_{j,N}},
\qquad w=(0.7,0.2,0.1).
```

三个损失依次是 Charbonnier、LPIPS、时间差分误差；sⱼ,N 在训练集上估计后固定，base 项不参与反向传播。**A 越大，表示恢复效果相对 base 越好。**

两种训练损失均最小化：

```math
L_{\mathrm{Softplus}}(A)=\tau\log(1+e^{-A/\tau}),\qquad \tau=0.1,
```

```math
L_{\mathrm{Linear}}(A)=-0.5A.
```

它们对 A 的导数是：

```math
\frac{dL_{\mathrm{Softplus}}}{dA}=-\frac{1}{1+e^{A/\tau}},
\qquad
\frac{dL_{\mathrm{Linear}}}{dA}=-0.5.
```

因此两者在 A=0 时梯度相同；A 增大后，Softplus 的梯度幅度减小，Linear 保持不变。

Softplus + Linear 的混合项可以写为下式，**本轮尚未验证**：

```math
L_{\mathrm{Hybrid}}(A)=L_{\mathrm{Softplus}}(A)-\lambda A,\qquad \lambda>0.
```

## 2. 实验结果

每组 **200 次更新、200 条训练生成轨迹**。REDS，9 帧、256×256，单个训练 seed；在 40 个非选模窗口上评估。表中数值均为相对原始 SeedVR 的变化。

| 方法 | NFE | 恢复 advantage ↑ | PSNR 增益 dB ↑ | LPIPS 减少 ↑ |
|---|---:|---:|---:|---:|
| Softplus | 4 | +0.3605 | +1.267 | +0.0238 |
| **Linear** | 4 | **+0.4192** | **+1.529** | +0.0130 |
| 旧 Flow-GRPO 适配版 | 4 | −0.0246 | −0.072 | −0.0024 |
| Softplus | 8 | +0.4188 | +1.575 | +0.0138 |
| **Linear** | 8 | **+0.4684** | **+1.843** | −0.0021 |
| 旧 Flow-GRPO 适配版 | 8 | −0.0337 | −0.108 | −0.0031 |

训练耗时：**Linear 28.7 分钟、Softplus 30.2 分钟、Flow-GRPO 7.4 分钟**，不含加载、验证和评估。匹配的是更新和采样数，不是 GPU 时间或 FLOPs。Flow-GRPO 为仅训练相同修正场的适配版。

![结果与置信区间](results/seedvr_linear_budget_v1/comparison.png)

## 3. 发现

1. **Linear 的最终恢复指标更高。** 相比 Softplus，PSNR 高约 0.26–0.27 dB；相比旧 Flow-GRPO，高约 1.60–1.95 dB。
2. **提升存在取舍。** Linear 的 LPIPS 比 Softplus 更差，特征核得分也更差，不能概括为所有质量指标都提高。
3. **Softplus 会衰减已经超过 base 的样本的梯度。** 这个现象可称为“正 advantage 的梯度饱和”。它没有显式限制输出与 base 的距离，因此不是 KL 正则或信赖域约束。
4. **目前不能说 Linear 收敛更快。** 第 50 次更新时 Softplus 领先，后续检查点 Linear 领先；两者最佳都在第 200 次更新。Hybrid 尚未验证。

**建议名称：奖励梯度饱和消融（Reward-gradient saturation ablation）。**

> 在固定更新和采样预算下，Linear 相比 Softplus 获得更高的最终恢复指标，但伴随感知指标和特征核得分退化。这与 Softplus 的正 advantage 梯度衰减一致，尚不能证明普遍的收敛加速。

本轮只有一个训练 seed，且这 40 个窗口此前已查看过，结论限于当前探索性实验。

记录：[学习曲线](results/seedvr_linear_budget_v1/validation_curves.png) · [完整指标与配对置信区间](results/seedvr_linear_budget_v1/metrics.json) · [选模记录](results/seedvr_linear_budget_v1/validation.json)
