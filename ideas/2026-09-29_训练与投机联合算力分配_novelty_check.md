# 在线训 draft 与投机解码的"统一算力分配"：novelty check 与判断

日期：2026-09-29（会话"算力分配投机"）
状态：未成型 idea（brainstorm）。本文件核查的是下面这段提案。

> 训练和服务抢算力：在线训练最该在低负载时跑，而低负载恰好也是投机收益最大的时候，高负载时两者都该让路。所以训练调度本身也应纳入 cost model，形成统一的"算力分配"问题：每个窗口 s_spec + s_train ≤ S(t)，最大化 即时增益(s_spec) + 未来增益(Δα(s_train))，是"消费 vs 投资"的跨期分配；训练价值取决于未来低负载时段占比，需要负载预测；高负载时只记录 token/top-k logits，低负载再训；真正冲突在 HBM；与 compute allocation 主线是同一结构的跨期扩展。

---

## 0. 一句话结论

**框架通顺，但两半和闭环都已经有人做了；这段话里有两处物理判断不对；与主线的"对得上"是表面的。** 唯一没被占的是"同卡共置的在线 draft 训练 + 懂'训练回报靠未来投机兑现'的调度器"。而且按它自己的前提（短上下文、高负载时投机关闭），最优策略几乎是平凡的，只有在它假设掉的情形（长上下文/MoE，或短碎的 API 型流量）里才真有权衡。**先花 2–3 天测余量，不要先写理论。**

---

## 1. 撞车地图

| 提案里的主张 | 已有工作 | 核实方式 |
|---|---|---|
| 低负载时用空闲算力训 draft | **OSD**（Liu et al., ICML 2024, arXiv 2310.07177）§4.2.1 原文："Depending on the system load and the rate at which the query distribution changes, users can adjust I accordingly. For example, we can perform a gradient update opportunistically only when the service traffic is not on spike (i.e., spare flops are available)." 附录 A.4 算了 draft/target FLOPs 比 18.75（160M draft），并称更新"can even be executed concurrently on separate devices" | PDF 原文 grep |
| 训练—服务闭环系统 | **Aurora**（Wang et al., Together AI + Tri Dao + Percy Liang 等, arXiv 2602.06932, v5 2026-08-04；代码 github.com/togethercomputer/aurora，MIT）：从线上 trace 持续学 speculator，SGLang 推理 + 异步训练服务器，热替换。B.1："we employ a split-GPU architecture … training the draft model on a separate GPU to avoid contention between serving and learning"；"updates triggered every 100 inference requests"；从零的 speculator "converges to 2.46 within approximately 10,000 requests"；"once the speculator is trained on-policy … simple token-level objectives capture most of the attainable gains"。README："Training and inference GPU sets … **must not overlap**"；repo 默认布局 train 0,1 / sglang 2,3,4,5（连 "all GPUs shared" 的 online 例子也是 4 推理 + 2 训练分开） | PDF 原文 + README 原文 |
| 同上（生产） | Together **ATLAS / ATLAS-2**：静态重 speculator + 在线学的轻 speculator + 置信度控制器；训练放在哪、是否按负载调度未公开 | 官方博客 |
| 在线学习理论 | **OnlineSPEC**（Qian, …, Hao Zhang, Peng Zhao, arXiv 2603.12617）：乐观在线学习、在线集成多 draft；无调度/资源模型 | 摘要 |
| 训练感知自投机 | **DVI**（arXiv 2510.05421）：verifier 接受/拒绝转成监督信号在线更新 draft head | 摘要 |
| 空闲 GPU 训 draft + 按 batch 选投机策略（RL） | **TLT**（Hu et al., ASPLOS 2026, arXiv 2511.16665）Spot Trainer：Worker Coordinator 监控 BUSY/IDLE/TRAINING，"Once the number of idle workers exceeds a configurable threshold, the coordinator promotes them to training mode"，可抢占 + 异步 checkpoint；Adaptive Rollout Engine 按 batch 选 SD 策略。**FastGRPO**（ICLR 2026, 2509.21792）并发感知投机 + 在线 draft 学习；**ReSpec**（2510.26475）；**Online Draft Co-Training**（2609.07108, 2026-09-07, 到 122B/256K）；EfficientRollout（2606.18967） | TLT PDF 原文；其余摘要 |
| 投机随负载开关/调长度 | TurboSpec（原 SmartSpec, 2406.14066）、FASER（2604.20503）、DSpark 调度器（2607.05147）、vLLM Adaptive Verification、SGLang 相关 PR——即 2026-08-31 判死的那条（见 project_specdecode_novelty） | 已有记录 |
| 训练塞进服务空闲并保 SLO | **FlexLLM**（NSDI 2026, 2402.18789）token 级共服务推理与 PEFT；**DeltaServe**（2607.28848, 2026-07-30）按延迟预测的调度器把空闲推理容量转成 LoRA 微调吞吐；Loquetier、LLMStation、OOCO。**它们训的任务与服务无关**，不形成闭环 | 摘要 |
| "低谷为未来投资 + 负载预测" 这个 framing | *Idleness is not sloth*（Golding et al., USENIX Winter 1995）：低利用率时段"eagerly performing work that might be needed in the future"；**QB5000**（Ma et al., SIGMOD 2018）预测查询到达率、提前建索引 | 检索确认 |

**剩下的空档**：同卡共置的在线 speculator 训练（不要专用训练卡）+ 价值感知调度（训练回报 = 未来开投机时的 Δα × 投机可用流量，扣掉当下的 SLO/KV 代价）。Aurora（分卡、定频）、FlexLLM/DeltaServe（共置但无闭环）、TLT（整卡空闲、RL 结构性长尾）各差一步。**并发风险高**：DeltaServe 把"微调任务"换成 draft 训练、或 Aurora 下一版加共置，都只差一步。

---

## 2. 技术论断逐条核对

**说对的：**
- 训练价值取决于未来还开不开投机——这个耦合是真的，也是前人没形式化的唯一一点。
- HBM 比 FLOPs 更硬（OSD 当年说"draft 小不占带宽"，那是 160M 独立 draft；EAGLE-3 头 ~0.4B，权重+梯度+Adam ≈ 6.4GB，另加激活）。
- "先实测共置干扰再决定建不建模"——方向对。

**说错/不精确的：**
1. **"高负载投机变负，两者都退出"只对短上下文 dense 成立。** MagicDec（2408.11049）：中长序列、batch 32–256 仍最高 2.51×；MoESD（2505.19645, NeurIPS 2025 spotlight）：中等 batch 下 MoE 比 dense 更受益，且越稀疏有效区间越宽。2026 主流负载（推理长输出 + MoE）正是这个情形。
2. **slack 不是标量，`s_spec + s_train ≤ S(t)` 写错了。** 投机吃的是 memory-bound 解码步里闲置的算术单元（权重读一次，多验几个 token 近乎免费），而且省时间；训练是另起的 kernel，要自己的时间片（或空分共享时抢 SM/带宽）+ 显存。低负载时投机反而给训练腾出空闲时间，两者不是切同一块饼。
3. **"draft 小、训练 FLOPs 不多"要打折。** OSD 的 18.75× 是 160M 独立 draft。EAGLE-3 训练要展开 ~7 步 TTT，粗估 8B 目标下每 token 训练 FLOPs 与目标推理每 token 同量级。训练开销是"在多少流量上训"的选择，不是天然小。
4. **"高负载记 logits、低负载再训"漏了两件事。** ① EAGLE 类还要存目标三层 hidden states（Aurora A.2 公式 T×3d×2 字节，d=4096 时约 24KB/token；logits 经 draft 词表 + top-1024 过滤后只剩 2KB，hidden states 成为主要开销）；② **on-policy 数据只在投机开着时产生**（draft 连猜 + 目标验证）。LongThink-Spec 自测（offcont−SFT chained +0.009 CI 含 0，双模型双域）与 Draft-OPD 都表明离线/teacher-forced 数据很快饱和，只有 on-policy 推得动链式接受率。高负载时记下的正是"便宜但会饱和"的那种。

---

## 3. 反直觉推论（最关键）

**在提案自己的前提下（短上下文，高负载投机关闭），它指出的耦合反而证明最优策略平凡**：高负载时训出的 draft 要到下一个低谷才用得上，而低谷时训练几乎免费，且线上更新几分钟内生效（Aurora 从零约 1 万请求收敛；增量漂移更快）；况且白天流量训出的 draft 未必适合夜间流量构成。→ "只在低谷训"（OSD 那一句话）≈ 最优，损失只是每个低谷开头的热身期。

**真正需要统一 cost model 的是它假设掉的两个情形：**
- **长上下文 / MoE**：高负载时投机仍开，更好的 draft 在峰值立刻有回报；但峰值训练要占 KV 和 TPOT。roofline 粗估（Llama-3.1-8B / H100，见 `code/2026-09-29_roofline_estimate.py`）：给训练 10GB，KV 池缩 18–23%，吞吐只掉 4.6–6.3%（memory-bound 区吞吐对 batch 不敏感）——与 draft 变好的增益同量级，是真权衡。
- **短碎、无周期的 API 型流量**：BurstGPT（2401.17644）图 2–5：对话服务有日/周周期，API 服务无周期。低谷短而碎时热身期占比变大，调度才有价值。讽刺点：**负载预测最容易的地方（周期性对话流量）最不需要预测。**

---

## 4. 与主线的关系

边际回报相等只是任何资源分配的 KKT 条件，不构成"同一框架"。主线（CAOPD/RAVTB/CSDIFF/BURSAR）的核心量是 consequence/题目级分配；这里是两种后台用途之间、以吞吐为目标的系统调度，没有 consequence 这个量。硬连会再挨"同一框架第 N 次搬家"。若做，它是一篇独立系统论文（NSDI/EuroSys/ASPLOS/OSDI；MLSys），评审只看 trace 回放收益与工程量。符合 9/25"可部署方法论文"规则，但工程重（SGLang 内共置训练 + 调度器 + trace 回放）。

---

## 5. Rung 0：先测余量（2–3 天，$0）

- **A 共置干扰（实验室服务器 A100，单卡）**：用 Aurora 开源代码，把训练进程与 SGLang 强行放同一张卡（README 禁止的那种），扫负载；测 P50/P99 TPOT、TTFT、KV 容量、训练吞吐；时间片 vs 低优先级 stream 两种共置。
- **B 适应曲线**：Aurora 的 ordered-stream 域切换协议，测切换后接受长度随训练样本数的饱和点。
- **C trace 回放模拟（本机 CPU）**：BurstGPT（对话 vs API）+ Azure 2024 一周 trace；步时模型取自 A，α(t) 取自 B；策略：静态 draft / Aurora 分卡 / OSD 阈值 / 一直共训 / 知道未来的 oracle（DP）。两个情形：短上下文 dense（Qwen3-8B）vs 长上下文或 MoE（Qwen3-30B-A3B）。泛化轴：≥2 模型家族、≥2 trace、2 个上下文区间。

**kill 标准：**
- K1：两个情形里 oracle 都只比最好的简单策略高 <5% goodput → 统一 cost model 没论文。
- K2：共置后 P99 TPOT 怎么都压不到 +10% 以内 → 需要 FlexLLM 级机制工程，重估成本。
- K3：与"一张专用训练卡分给 N 个副本"比，共置只在 N≈1 时占优 → 场景收窄为单节点私有部署（窄但成立）。注意 Aurora A.3 已支持多个推理服务器向同一个训练服务器汇聚数据，所以"分摊后的专用训练卡"是现成的强基线。

**若通过，论文形状**："不要专用训练卡的 Aurora"——同卡共置在线 speculator 学习 + 价值感知调度（长上下文/MoE 下与 KV 抢显存的权衡是核心），主表 = 真实 trace 下 SLO 达成率 / goodput / 每 token 成本，对比 Aurora（分卡）、OSD（阈值）、TurboSpec（只调投机）、FlexLLM/DeltaServe 式通用共置。

**资源注意**：Vista ECS25019 的 SU 9/30 过期且是 aarch64（SGLang/Aurora 依赖多半要源码编）；OSC 未经许可不计费；实验室服务器 GPU0 被别的项目常驻，A 用 GPU1（开跑前确认空闲）。

---

## 6. 来源

- OSD: https://arxiv.org/abs/2310.07177
- Aurora: https://arxiv.org/abs/2602.06932 ｜ https://github.com/togethercomputer/aurora
- ATLAS: https://www.together.ai/blog/adaptive-learning-speculator-system-atlas
- OnlineSPEC: https://arxiv.org/abs/2603.12617
- DVI: https://arxiv.org/abs/2510.05421
- TLT: https://arxiv.org/abs/2511.16665
- FastGRPO: https://arxiv.org/abs/2509.21792
- ReSpec: https://arxiv.org/abs/2510.26475
- Online Draft Co-Training: https://arxiv.org/abs/2609.07108
- TurboSpec: https://arxiv.org/abs/2406.14066 ｜ FASER: https://arxiv.org/abs/2604.20503
- FlexLLM: https://arxiv.org/abs/2402.18789 ｜ DeltaServe: https://arxiv.org/abs/2607.28848
- MagicDec: https://arxiv.org/abs/2408.11049 ｜ MoESD: https://arxiv.org/abs/2505.19645
- BurstGPT: https://arxiv.org/abs/2401.17644
- Idleness is not sloth: https://www.usenix.org/conference/usenix-1995-technical-conference/idleness-not-sloth
- QB5000: https://dl.acm.org/doi/10.1145/3183713.3196908
