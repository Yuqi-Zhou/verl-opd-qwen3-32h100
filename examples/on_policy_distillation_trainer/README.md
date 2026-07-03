# On-Policy Distillation

This trainer jointly trains a student model with policy-gradient on-policy rollouts and a distillation loss against a frozen teacher model served by a separate Ray cluster. Compared to pure SFT from teacher generations, on-policy distillation typically closes more of the teacher/student gap at the same compute budget.

## Canonical Scripts

| Script                          | Teachers | Modality   | Infer | Train    | Platform |
|---------------------------------|----------|------------|-------|----------|----------|
| `run_qwen3_8b_fsdp.sh`          | single   | text       | vLLM  | FSDP     | NVIDIA   |
| `run_qwen3_8b_megatron.sh`      | single   | text       | vLLM  | Megatron | NVIDIA   |
| `run_qwen3_vl_8b_fsdp.sh`       | single   | VL         | vLLM  | FSDP     | NVIDIA   |
| `run_qwen3_8b_mopd_fsdp.sh`     | multi    | text + VL  | vLLM  | FSDP     | NVIDIA   |
| `run_qwen3_30b_a3b_235b_moe_fully_async_32h100.sh` | single | text | vLLM | Megatron | 32×H100 fully_async OPD |

### 32×H100 aggressive OPD (Qwen3-30B-A3B + Qwen3-235B-MoE)

Long-context (18K+2K) recipe with three-pool GPU layout and `fully_async_main`.
See **[CODEX_README.md](../../CODEX_README.md)** for Codex / cluster deployment guide.

```bash
cp examples/on_policy_distillation_trainer/env_32h100.example .env.32h100
# edit paths, then:
set -a && source .env.32h100 && set +a
OPD_PHASE=conservative bash examples/on_policy_distillation_trainer/run_qwen3_30b_a3b_235b_moe_fully_async_32h100.sh
```

Override `STUDENT_MODEL` and `TEACHER_MODEL` via env vars to swap model pairs in
the single-teacher scripts. The MOPD script exposes per-teacher overrides.

## Key Flags

- `distillation.enabled=True`
- `distillation.teacher_models.teacher_model.model_path=<HF path>` (single-teacher)
- `+distillation.teacher_models.<name>.{key,model_path,num_replicas,inference.*}` (multi-teacher)
- `distillation.distillation_loss.loss_mode={k1, k3, forward_kl_topk, ...}`
- `distillation.distillation_loss.use_policy_gradient=True|False`
- `distillation.distillation_loss.topk=64`
