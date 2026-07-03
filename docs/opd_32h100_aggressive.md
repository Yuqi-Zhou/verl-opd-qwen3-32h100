# 32×H100 OPD Aggressive Recipe

This document is a short pointer. **Codex agents should read [CODEX_README.md](../CODEX_README.md) first.**

## Quick links

- Launch script: `examples/on_policy_distillation_trainer/run_qwen3_30b_a3b_235b_moe_fully_async_32h100.sh`
- Env template: `examples/on_policy_distillation_trainer/env_32h100.example`
- Upstream OPD docs: `docs/algo/opd.md`
- fully_async 30B benchmark: `verl/experimental/fully_async_policy/README_zh.md`

## GPU layout (32 cards)

```text
8 rollout + 8 train + 16 teacher = 32
```

## Phases

```bash
OPD_PHASE=conservative  # smoke test
OPD_PHASE=aggressive    # max throughput (default)
```
