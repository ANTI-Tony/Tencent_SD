# weight_edit_draft · Rung 0：MIT 权重编辑版 draft 在 greedy 投机解码里的接受长度

**问题**：MIT"权重空间编程"把 Qwen3.5-0.8B 的 MATH-500 准确率从 7.9% 改到 48.1%（只改两层 down_proj 的 131,072 个权重）。
把这个"更准"的小模型当 draft，给同族 9B target 做 greedy 投机解码，接受长度会不会变？这是"大模型改小模型权重→接受更多"
这条假设最干净的检验：准确率变了、其余全没变。

**预注册预测（2026-11-04，跑前写下）**
| draft | 相对原版 0.8B 的每轮接受草稿 token 数 |
|---|---|
| MIT 编辑版 0.8B（p50c） | ±2% 以内（去循环方向只在 draft 自己的长轨迹里激活，target 前缀上不出现） |
| Qwen3.5-2B（能力型参照） | 明显更高 |

判据：编辑版相对原版 > +5% 且配对 bootstrap 95% 下界 > 0 → 预测错误，继续下一步（量"空间"：偏置校准下界 vs LoRA 蒸馏上界）；
否则"准确率 ≠ 被接受"成立，编辑路线在无损投机解码里到此为止。

**设计**
- target：Qwen/Qwen3.5-9B，greedy，thinking on，输出上限 2048 token；用户轮 = 题目 + MIT 同款后缀（"Please reason step by step, and put your final answer within \boxed{}."）。
- 题目：MATH-500 测试集固定种子抽 256 题（`data/math500_rung0_256.jsonl`，seed 0；MIT 练习集只用 GSM8K/MATH 训练集，无泄漏）。
- draft：Qwen/Qwen3.5-0.8B（revision 2fc0636…，MIT 编辑的底座）、RikiyaT/Qwen3.5-0.8B-weight-space-programming（p50c 编辑，bit-exact）、Qwen/Qwen3.5-2B。
  四个模型 tokenizer.json 哈希相同（5f9e4d4901a9…），chat 模板只差 enable_thinking 的默认值（0.8B/2B 默认关、4B/9B 默认开），脚本显式传参。
- **精确核算，不跑投机引擎**：greedy + target 固定时，被接受的 token 恒等于 target 的贪心轨迹。记 a_t = [draft 在 target 轨迹第 t 位的 teacher-forced argmax == target token]，
  每轮接受数 = 从该位起连续 1 的长度截到 γ=5，然后 target 补 1 个 token（纠正或奖励）。`analyze.simulate` 就是这个过程；`draft_agree.py` 只做一次 teacher-forced 前向。
- 诊断：首拒位置分布（第 1…5 个）、首拒 token 类别（数字/LaTeX/空白/词…）、分歧是否"复制型"（提议与前 3 个 token 构成的 4-gram 已在上文出现）、
  编辑版"修好"与"弄坏"的位置里复制型占比（直接回答"少掉的拒绝是否集中在重复片段"）。

**文件**
- `common.py` 模板/模型加载；`gen_target.py` target 轨迹（vLLM，HF 备用）；`draft_agree.py` draft teacher-forced；`analyze.py` 核算与表格；
  `verify_mit_edit.py` 核对 HF 编辑版与底座只差两个张量且 sha256 与 MIT manifest 一致。
- `jobs/env.sh`（OSC Cardinal，复用 specsched 的 ss312 环境：vLLM 0.27.1 / transformers 5.15.1）、`jobs/prefetch.sh`（登录节点下载权重，不计费）、`jobs/rung0.sbatch`（debug 分区 1×H100 ≤1 h）。
- 本机 CPU 冒烟测试：`runs/local/`（Qwen2.5-0.5B 自己当 target 和 draft，一致率 1.000，表格正常）。

**跑法（OSC）**
```
sbatch jobs/rung0.sbatch            # 默认 RUN=r0 TARGET=Qwen/Qwen3.5-9B MAXNEW=2048 THINK=1
tail -f /fs/scratch/PCS0296/specsched/logs/wed-rung0-<jobid>.out
cat /fs/scratch/PCS0296/specsched/code/weight_edit_draft/runs/r0/summary.md
```
