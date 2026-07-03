# OPD 32×H100 Skill

When implementing or deploying on-policy distillation for **Qwen3-30B-A3B + Qwen3-235B-MoE** on **32×H100**:

1. Read **[CODEX_README.md](../../CODEX_README.md)** in the repository root first.
2. Use launch script: `examples/on_policy_distillation_trainer/run_qwen3_30b_a3b_235b_moe_fully_async_32h100.sh`
3. Start with `OPD_PHASE=conservative`, then `OPD_PHASE=aggressive`.
4. Do **not** switch to `forward_kl_topk` for 18K+2K context without explicit user request (OOM risk).
5. Keep three-pool layout: 8 rollout + 8 train + 16 teacher GPUs.
