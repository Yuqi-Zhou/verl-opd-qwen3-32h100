# OPD 32×H100 激进方案 — Codex 实施指南

> **给 Codex / 后续 Agent 的说明**：本仓库在 verl `0.8.0.dev` 基础上，新增了 **Qwen3-30B-A3B student + Qwen3-235B-MoE teacher + 18K/2K 长上下文** 的激进 OPD 训练配方。请 **以本仓库为参考**，将相同配置迁移到目标训练集群（SDP / Merlin / 自建 Ray 集群）。

---

## 1. 仓库是什么

| 项 | 值 |
|----|-----|
| 基线 | verl `0.8.0.dev`（含 Megatron OPD response_mask 优化，**非** `-clean` 版） |
| 路径 | `~/Projects/verl-opd-qwen3-32h100` |
| 核心新增 | 32 卡三池 fully_async OPD 启动脚本 + 本文档 |
| 未改 verl 核心代码 | 仅新增 `examples/` 脚本与文档；框架逻辑仍走 upstream verl |

---

## 2. 你应该复制什么到目标环境

### 必复制（最小集）

```text
examples/on_policy_distillation_trainer/run_qwen3_30b_a3b_235b_moe_fully_async_32h100.sh
examples/on_policy_distillation_trainer/env_32h100.example
CODEX_README.md   # 本文件
```

### 建议一并复制（完整 verl）

整个仓库 `rsync` 到训练节点，或 `pip install -e .` 安装本 fork。

```bash
rsync -av --exclude='.git' ~/Projects/verl-opd-qwen3-32h100/ user@cluster:/path/to/verl-opd-qwen3-32h100/
```

---

## 3. 硬件与 GPU 切分（32×H100，4 节点×8 卡）

```text
Worker0  [8 GPU]  Rollout Pool   student vLLM async   (2 replicas × TP4)
Worker1  [8 GPU]  Train Pool     student Megatron     (TP2 × CP2 × EP2)
Worker2  [8 GPU]  Teacher Pool   teacher replica-0    (TP8)
Worker3  [8 GPU]  Teacher Pool   teacher replica-1    (TP8)
```

**约束**（distillation config 校验）：

```text
teacher_pool = distillation.nnodes × distillation.n_gpus_per_node = 16
teacher_footprint = num_replicas × tensor_model_parallel_size = 2 × 8 = 16  ✓
total = rollout(8) + train(8) + teacher(16) = 32  ✓
```

Ray 需保证 **teacher 的 16 卡落在 Worker2–3**，rollout 在 Worker0，train 在 Worker1。若 Ray 自动调度不符合，需在集群侧配置 placement group / 节点标签。

---

## 4. 一键启动

```bash
cd /path/to/verl-opd-qwen3-32h100

# 复制并编辑环境变量
cp examples/on_policy_distillation_trainer/env_32h100.example .env.32h100
# 修改 STUDENT_MODEL / TEACHER_MODEL / TRAIN_FILE / TEST_FILE / RAY_ADDRESS

set -a && source .env.32h100 && set +a

# 阶段 A：先保守冒烟（staleness=0, gen_batch=1）
OPD_PHASE=conservative bash examples/on_policy_distillation_trainer/run_qwen3_30b_a3b_235b_moe_fully_async_32h100.sh

# 阶段 B：激进吞吐（staleness=0.5, partial_rollout=True）
OPD_PHASE=aggressive bash examples/on_policy_distillation_trainer/run_qwen3_30b_a3b_235b_moe_fully_async_32h100.sh
```

---

## 5. 设计决策摘要（Codex 改配置时勿随意推翻）

| 决策 | 取值 | 原因 | 来源 |
|------|------|------|------|
| 训练入口 | `fully_async_main` | 流水线 rollout/train，提高 GPU 利用率 | `fully_async_policy/README_zh.md` 30B 1.7× 实验 |
| `hybrid_engine` | `false` | 三池硬分离 | OPD e2e + 本方案 |
| Loss | `k1` + `use_policy_gradient=true` | 20K 上下文不物化 full vocab logits | Issue #6810, PR #6593 |
| **不用** `forward_kl_topk` | — | 长 context OOM 仍 open | Issue #6810 |
| `bypass_mode` | `true` | 跳过 `old_log_prob` 重算 | `fully_async_ppo_megatron_trainer.yaml` 默认 |
| Student backend | Megatron MoE | TP2×CP2×EP2，CP 切 20K 序列 | `grpo_30b_a3b_base_math_megatron_96_32.sh` |
| Teacher | standalone vLLM pool | 不与 student 抢卡 | PR #5723, #5745 |
| `max_logprobs=1` | k1 只需 sampled token | 减 teacher 通信与计算 | `docs/algo/opd.md` |
| `checkpoint_engine` | nccl | 30B 权重 sync 4.38s vs 15.76s | fully_async README |

---

## 6. 分阶段调参（跑通后再拧）

脚本通过 `OPD_PHASE` 切换：

| 参数 | conservative | aggressive |
|------|--------------|------------|
| `OPD_PHASE` | `conservative` | `aggressive` |
| `gen_batch_size` | 1 | 4 |
| `staleness_threshold` | 0.0 | 0.5 |
| `partial_rollout` | False | True |
| `trigger_parameter_sync_step` | 1 | 4 |
| `require_batches` | 1 | 2 |
| `teacher max_num_batched_tokens` | 4096 | 8192 |

**激进阶段顺序**（每次只改一项，观察 20 step）：

1. `OPD_PHASE=aggressive`
2. `TEACHER_MAX_BATCHED_TOKENS=12288`（teacher OOM 则回退）
3. `GEN_BATCH=8`
4. `STALENESS=0.5` 已默认；可试 `0.6`

---

## 7. 监控指标与产物落盘

### Loss（console / TensorBoard / WandB）

启动脚本默认：

- `trainer.logger='["console","tensorboard","wandb"]'`
- `export TENSORBOARD_DIR=${CKPT_ROOT}/tensorboard`（由 `verl.utils.tracking` 读取）
- 关键标量：`actor/distillation/loss`、`training/rollout_actor_probs_pearson_corr`

### Checkpoint 与 Rollout

| 配置 | 默认 | 说明 |
|------|------|------|
| `SAVE_FREQ` | `20` | fully_async 按 **`current_param_version`** 存盘 |
| `DEFAULT_LOCAL_DIR` | `${CKPT_ROOT}/checkpoints` | actor checkpoint |
| `ROLLOUT_DATA_DIR` | `${CKPT_ROOT}/rollouts` | 每步 rollout jsonl |
| `RESUME_MODE` | `auto` | 断点续训 |

**GitHub → 服务器部署**见：`CODEX_GITHUB_AND_SERVER_HANDOFF.md`

### 健康指标

| 指标 | 健康 | 异常处理 |
|------|------|----------|
| `training/rollout_actor_probs_pearson_corr` | > 0.95 | 检查 bypass、vLLM vs Megatron 数值；Qwen3.5 查 `use_remove_padding` |
| Rollout GPU util | > 70% | 增大 `GEN_BATCH` 或 rollout 卡数 |
| Train GPU util | 50–80% | async 正常；长期过低说明 rollout/teacher 慢 |
| `actor/distillation/loss` | 有限、不 NaN | 降 lr、检查 clamp |
| Teacher OOM | — | 降 `TEACHER_MAX_BATCHED_TOKENS` → 4096 → 2048 |

---

## 8. Teacher OOM 降级 ladder

```bash
export TEACHER_MAX_BATCHED_TOKENS=4096
export TEACHER_MAX_SEQS=1
# 仍 OOM：需加 teacher 卡（3×TP8=24 卡）并相应减少 rollout 卡
```

参考：[Issue #6792](https://github.com/verl-project/verl/issues/6792)（235B teacher OOM，仍 open）

---

## 9. Codex 在目标集群上的实施 Checklist

```text
[ ] 1. rsync 本仓库到集群
[ ] 2. pip install -e . （或按集群现有 verl 安装方式）
[ ] 3. 确认 4 节点×8 卡 Ray 集群就绪，设置 RAY_ADDRESS
[ ] 4. 节点绑定：Worker0=rollout, Worker1=train, Worker2-3=teacher
[ ] 5. 准备 STUDENT_MODEL / TEACHER_MODEL 权重（同 tokenizer 族）
[ ] 6. 准备 parquet 数据，prompt 截断到 18000
[ ] 7. OPD_PHASE=conservative 跑 20 step 冒烟
[ ] 8. 检查 pearson_corr、无 OOM
[ ] 9. OPD_PHASE=aggressive 正式训练
[ ] 10. 按第 6 节逐步调参
```

---

## 10. 若目标环境不是 32 卡

按比例缩放，保持 **teacher 占 50%**、rollout 25%、train 25%：

| 总卡数 | Teacher | Rollout | Train |
|--------|---------|---------|-------|
| 32 | 16 | 8 | 8 |
| 64 | 32 | 16 | 16 |
| 128 | 64 | 32 | 32 |

修改脚本中的 `NGPUS_*` / `TEACHER_*` 环境变量即可；**teacher replicas × TP 必须等于 teacher pool 大小**。

---

## 11. 相关 upstream 参考

- verl OPD 文档：`docs/algo/opd.md`
- fully_async 30B 实验：`verl/experimental/fully_async_policy/README_zh.md`
- OPD e2e 测试脚本：`tests/special_e2e/run_fully_async_policy_opd.sh`
- 30B Megatron async：`verl/experimental/fully_async_policy/shell/grpo_30b_a3b_base_math_megatron_96_32.sh`

---

## 12. 联系上下文

本配方来自 Cursor Agent 对以下问题的分析：

1. OPD 调度：student 等 teacher（sync barrier）
2. Student 推理慢（多次 forward）
3. Update OOM（`forward_kl_topk` + 长 context）

若需回溯设计 rationale，见对话记录中的三池 + fully_async + k1 方案说明。
