# Tencent_SD：在线训练草稿模型 × 投机解码的统一算力分配（实验代码）

这里只有实验代码、数据处理代码和 Markdown 文档。论文源码、论文用的出表出图脚本、画图脚本和原始运行数据都不在这里。

## 目录

| 路径 | 内容 |
|---|---|
| `code/specsched/` | 控制器、实验 harness、同卡训练器、特征导出、各实验矩阵生成器、分析与重放脚本 |
| `code/specsched/sched/` | 冻结模拟器 `sim.py`、校准文件 `model*.json`、留出验证入口 `validate_heldout.py`、第 10 项补充 `sim_arm.py` |
| `code/specsched/jobs/` | OSC 上的 Slurm 作业脚本（不含环境脚本） |
| `code/specsched/sgl/` | SGLang 移植 |
| `code/vllm_pr/` | vLLM 引擎扩展补丁与说明 |
| `code/weight_edit_draft/` | 另一条探索线（权重编辑草稿、草稿词表扩展），不进论文 |
| `ideas/` | 计划与预注册 |
| `results/summaries/` | 各阶段结果总结，含 E28 同起点对照与新版证据核查回复 |

## 冻结检查点的文件哈希

SHA-256 前 16 位，与论文表 6 一致。

| 文件 | 哈希 |
|---|---|
| `code/specsched/controller.py` | `0c526960ecb25143` |
| `code/specsched/train_head.py` | `cc224db0c6b40c81` |
| `code/specsched/hotswap_run.py` | `866e1159bda44604` |
| `code/specsched/livetap.py` | `719538857da7d833` |
| `code/specsched/sched/model.json` | `18c3723adbe9b858` |
| `code/specsched/sched/model_m2.json` | `5c9fa2d452ca7156` |
| `code/specsched/sched/sim.py` | `451c5673bbb4c7a4` |
| `code/specsched/sched/validate_heldout.py` | `3bc93d6d86f35161` |

## 入口

| 实验 | 入口 |
|---|---|
| 冻结主矩阵（81 次） | `abl_matrix.py` 生成清单，`abl_runner.py` 领取运行，`jobs/abl.sbatch` 提交；单次运行是 `hotswap_run.py` |
| 混合流量与标签变体 | `abl_matrix_mix.py`、`fit_mix_model.py` |
| 消融 | `abl_matrix_ablate.py`、`controller_ablate.py`、`hotswap_run_ablate.py` |
| TIDE 式基线 | `abl_matrix_tide.py`、`abl_matrix_tidevar.py`、`controller_tide.py`、`hotswap_run_tide.py` |
| SGLang | `abl_matrix_sgl.py`、`hotswap_run_sgl.py`、`sgl_*.py`、`sgl/` |
| E28 同起点对照 | `abl_matrix_arm.py`、`controller_arm.py`、`hotswap_run_arm.py`；分析 `analyze_arm.py`、`scan_anomaly.py` |
| 模拟器留出验证 | `sched/validate_heldout.py`；它的 `main()` 会覆盖冻结结果，复现请用 `sched/sim_arm.py` |
| 决策重放 | `replay_value_baseline.py`、`replay_duty_compare.py`、`replay_switch_clause.py`、`replay_arm_value.py`、`replay_invert_axis.py` |
| 其余核对 | `frontier_check.py`、`interference_check.py`、`plan_rows_tables.py`、`goodput_check.py`、`output_equivalence.py` |

## 不在这里的东西

- 原始运行数据：每次运行的日志与逐请求输出，共约 30 GB，在 OSC 与本地归档。
- 作业环境脚本 `jobs/env.sh`、`jobs/env_ss312.sh`：只设置缓存路径与运行环境变量，按自己的集群改写即可。
- 结果表格与 Excel 汇总。
- `analyze_abl.py`、`analyze_c1.py`、`analyze_c2.py` 在输出表格之外会顺带保存诊断图，这部分保留未删。
