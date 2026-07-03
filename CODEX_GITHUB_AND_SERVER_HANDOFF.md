# Codex 交接：代码审阅 → GitHub 推送 → 服务器部署 → 启动训练

> **读者**：在浏览器已登录 GitHub 的 Codex / 另一台 Agent。  
> **目标**：把本机 `verl-opd-qwen3-32h100` 推到 GitHub，在 32×H100 训练服务器上 `git clone` 替换旧 verl，跑完整 OPD 训练（**记录 loss、保存 rollout、每 20 step 存 checkpoint**）。

---

## 0. 背景（给接手 Codex 的一句话）

在 **4 节点 × 8 H100 = 32 GPU** 上，用 **fully_async OPD** 蒸馏：

| 角色 | 模型 | GPU |
|------|------|-----|
| Student（rollout + train） | **Qwen3-30B-A3B** | 8 rollout + 8 train |
| Teacher | **Qwen3-235B-MoE** | 16（2×TP8） |

- **上下文**：`prompt=18000`, `response=2048`
- **Loss**：`k1` + `use_policy_gradient=true`（避免 `forward_kl_topk` 长上下文 OOM）
- **入口**：`verl.experimental.fully_async_policy.fully_async_main`
- **基线代码**：verl `0.8.0.dev`（含 Megatron OPD response_mask 优化）

本仓库 **未大改 verl 核心**，主要新增启动脚本、环境模板和本文档。

---

## 1. 请你先做的：代码审阅

审阅重点（按优先级）：

1. **`examples/on_policy_distillation_trainer/run_qwen3_30b_a3b_235b_moe_fully_async_32h100.sh`**
   - GPU 切分：8 rollout / 8 train / 16 teacher
   - `SAVE_FREQ=20` → `trainer.save_freq=20`（fully_async 里按 **`current_param_version`** 触发，不是 `global_steps`；`trigger_parameter_sync_step=4` 时约每 4 个 trainer micro-step 涨 1 个 param_version）
   - `trainer.rollout_data_dir` → 每步异步 dump rollout JSON（继承 `SeparateRayPPOTrainer._fit_dump_data`）
   - `trainer.default_local_dir` → checkpoint 目录
   - `TENSORBOARD_DIR` 环境变量 → TensorBoard loss 曲线
   - `trainer.logger` 含 `console` + `tensorboard` + `wandb`

2. **`examples/on_policy_distillation_trainer/env_32h100.example`**
   - 服务器路径占位是否清晰（`/mnt/bn/...`）
   - `STUDENT_MODEL` / `TEACHER_MODEL` 需用户在服务器上改成真实 checkpoint

3. **`CODEX_README.md`**
   - 设计 rationale、分阶段 `OPD_PHASE=conservative|aggressive`

4. **风险点**（审阅时标注，不必阻塞 push）：
   - Teacher 235B 在 18K prompt 下可能 OOM → 脚本内有 `TEACHER_MAX_BATCHED_TOKENS` 降级 ladder
   - Ray placement：teacher 16 卡需在 Worker2–3，需集群侧确认
   - Issue #6810：`forward_kl_topk` 仍不可用；当前配方已避开

审阅结论写进 PR description 或 commit message 即可。

---

## 2. 推送到 GitHub（浏览器操作，本机无 `gh` CLI）

### 2.1 在本机仓库目录

```bash
cd ~/Projects/verl-opd-qwen3-32h100
git status
git add -A
git commit -m "Add 32xH100 fully-async OPD recipe with loss logging and checkpoint/rollout saves"
```

### 2.2 在 github.com 创建仓库

1. 打开 https://github.com/new
2. Repository name 建议：`verl-opd-qwen3-32h100`（或用户指定）
3. **Private**（含内部路径信息）
4. **不要**勾选 “Add a README”（本地已有）
5. Create repository

### 2.3 推送（终端或 Codex 在本机执行）

把 `YOUR_USER` 换成实际 GitHub 用户名：

```bash
cd ~/Projects/verl-opd-qwen3-32h100
git remote add origin https://github.com/YOUR_USER/verl-opd-qwen3-32h100.git
git branch -M main
git push -u origin main
```

若需浏览器登录授权，按 Git Credential / SSO 提示完成。

**记下仓库 URL**，服务器部署要用：`https://github.com/YOUR_USER/verl-opd-qwen3-32h100.git`

---

## 3. 服务器环境（已知信息，部署前核对）

| 项 | 路径 / 值 |
|----|-----------|
| Python | `/opt/tiger/verl_envs/v080_qwen35_cu128/bin/python` |
| 旧 verl | `/opt/tiger/verl` |
| 环境脚本（若有） | `/opt/tiger/verl_q35_env.sh` |
| 输出根目录 | `/mnt/bn/search-nlp-vagcp/lvjiajun.a/RL/opd_runs` |
| 历史 student（Qwen3.5，仅供参考） | `.../model/answer_qwen3_5_35b_a10b_sft_v3_5/.../checkpoint-100` |
| 历史 teacher（122B，仅供参考） | `.../model/answer_qwen3_5_122b_a10b_sft_rlv3_full_refusal_.../checkpoint-1200` |

**本次目标模型**（路径需在服务器上确认后写入 `.env.32h100`）：

- Student: **Qwen3-30B-A3B**
- Teacher: **Qwen3-235B-MoE**

---

## 4. 服务器部署步骤

### 4.1 Clone 并安装

```bash
export VERL_INSTALL_DIR=/opt/tiger/verl-opd-qwen3-32h100
export PYTHON_BIN=/opt/tiger/verl_envs/v080_qwen35_cu128/bin/python

git clone https://github.com/YOUR_USER/verl-opd-qwen3-32h100.git ${VERL_INSTALL_DIR}
cd ${VERL_INSTALL_DIR}

# 可选：替换旧 verl 软链
ln -sfn ${VERL_INSTALL_DIR} /opt/tiger/verl

${PYTHON_BIN} -m pip install -e . --no-deps   # 若集群已装齐依赖
```

或使用仓库脚本：

```bash
bash scripts/server/deploy_from_github.sh https://github.com/YOUR_USER/verl-opd-qwen3-32h100.git
```

### 4.2 环境检查

```bash
cd ${VERL_INSTALL_DIR}
bash scripts/server/check_env_32h100.sh
bash scripts/preflight_opd_32h100.sh
```

### 4.3 配置 `.env.32h100`

```bash
cp examples/on_policy_distillation_trainer/env_32h100.example .env.32h100
vim .env.32h100   # 改 STUDENT_MODEL, TEACHER_MODEL, TRAIN_FILE, TEST_FILE, RAY_ADDRESS
```

---

## 5. 启动训练（两阶段）

```bash
cd ${VERL_INSTALL_DIR}
set -a && source .env.32h100 && set +a

# 阶段 A：冒烟 ~20 param_version
OPD_PHASE=conservative bash examples/on_policy_distillation_trainer/run_qwen3_30b_a3b_235b_moe_fully_async_32h100.sh

# 阶段 B：完整训练（loss + rollout + 每 20 step 存盘）
OPD_PHASE=aggressive bash examples/on_policy_distillation_trainer/run_qwen3_30b_a3b_235b_moe_fully_async_32h100.sh
```

---

## 6. 如何查看 Loss / 产物

### Loss 指标（console / TB / wandb）

| 指标 | 含义 |
|------|------|
| `actor/distillation/loss` | OPD 主 loss |
| `actor/pg_loss` | policy gradient 项 |
| `training/rollout_actor_probs_pearson_corr` | rollout vs train 数值一致性（应 > 0.95） |

### TensorBoard

```bash
tensorboard --logdir ${TENSORBOARD_DIR} --port 6006 --bind_all
# 默认: ${CKPT_ROOT}/tensorboard
```

### Checkpoint（每 `SAVE_FREQ=20` param_version）

```text
${DEFAULT_LOCAL_DIR}/
  global_step_20/
  global_step_40/
  ...
```

### Rollout dump（每训练 step）

```text
${ROLLOUT_DATA_DIR}/
  rollout_step_*.jsonl   # prompts / responses / scores
```

---

## 7. 完整 Checklist

```text
[ ] Codex 审阅 launch 脚本 + env 模板
[ ] git commit + push 到 GitHub
[ ] 服务器 git clone 到 /opt/tiger/verl-opd-qwen3-32h100
[ ] pip install -e . 或软链替换 /opt/tiger/verl
[ ] check_env_32h100.sh + preflight 通过
[ ] .env.32h100 填好模型与数据路径
[ ] Ray 4×8 就绪，节点角色 rollout/train/teacher 正确
[ ] OPD_PHASE=conservative 冒烟
[ ] TensorBoard / wandb 能看到 actor/distillation/loss
[ ] rollout 目录有 jsonl，checkpoint 每 20 step 出现
[ ] OPD_PHASE=aggressive 长跑
```

---

## 8. 关键文件索引

| 文件 | 用途 |
|------|------|
| `CODEX_README.md` | 方案设计与调参 |
| `CODEX_GITHUB_AND_SERVER_HANDOFF.md` | 本文件：GitHub + 服务器 |
| `examples/.../run_qwen3_30b_a3b_235b_moe_fully_async_32h100.sh` | 主启动脚本 |
| `examples/.../env_32h100.example` | 服务器环境变量模板 |
| `scripts/server/deploy_from_github.sh` | 一键 clone + install |
| `scripts/server/check_env_32h100.sh` | 环境探测 |
| `scripts/preflight_opd_32h100.sh` | 本地/服务器预检 |

---

## 9. 与用户沟通

推送完成后，把 **GitHub 仓库 URL** 和 **服务器上 `CKPT_ROOT` 路径** 发给用户；训练启动后提供 TensorBoard 端口或 wandb run 链接。
