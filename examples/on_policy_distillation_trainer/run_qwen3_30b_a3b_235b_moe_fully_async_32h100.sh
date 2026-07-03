#!/usr/bin/env bash
# Aggressive OPD recipe for 32xH100 (4 nodes x 8 GPUs):
#   Student: Qwen3-30B-A3B (Megatron train + vLLM async rollout)
#   Teacher: Qwen3-235B-MoE (standalone vLLM teacher pool)
#   Context: 18000 prompt + 2048 response tokens
#
# GPU layout (32 GPUs total):
#   Worker0 (8 GPU): Rollout pool   — student vLLM async, 2 replicas x TP4
#   Worker1 (8 GPU): Train pool      — student Megatron, TP2 x CP2 x EP2
#   Worker2-3 (16 GPU): Teacher pool — 2 replicas x TP8
#
# Entry: fully_async_main (pipeline rollout / teacher / train)
# Loss:  k1 PG-OPD (no full-vocab logits; safe at 20K context)
#
# Usage:
#   export STUDENT_MODEL=/path/to/Qwen3-30B-A3B
#   export TEACHER_MODEL=/path/to/Qwen3-235B-MoE
#   export TRAIN_FILE=/path/to/train.parquet
#   export TEST_FILE=/path/to/test.parquet
#   bash examples/on_policy_distillation_trainer/run_qwen3_30b_a3b_235b_moe_fully_async_32h100.sh
#
# Phases (set OPD_PHASE):
#   conservative — first smoke / stability (staleness=0, gen_batch=1)
#   aggressive   — max throughput (default; staleness=0.5, partial_rollout)
#
# See CODEX_README.md for full documentation.

set -xeuo pipefail

export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# NCCL workaround for some PCIe topologies (safe on NVLink H100 clusters too)
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export NCCL_CUMEM_HOST_ENABLE="${NCCL_CUMEM_HOST_ENABLE:-0}"

############################ Models & Data ############################

STUDENT_MODEL=${STUDENT_MODEL:-"/path/to/Qwen3-30B-A3B"}
TEACHER_MODEL=${TEACHER_MODEL:-"/path/to/Qwen3-235B-MoE"}
TRAIN_FILE=${TRAIN_FILE:-"/path/to/train.parquet"}
TEST_FILE=${TEST_FILE:-"/path/to/test.parquet"}
PYTHON_BIN=${PYTHON_BIN:-python3}
RAY_ADDRESS=${RAY_ADDRESS:-}

############################ Sequence Length ############################

MAX_PROMPT=${MAX_PROMPT:-18000}
MAX_RESPONSE=${MAX_RESPONSE:-2048}
MAX_LEN=$((MAX_PROMPT + MAX_RESPONSE + 1))

############################ 32-GPU Three-Pool Layout ############################

NNODES_ROLLOUT=${NNODES_ROLLOUT:-1}
NGPUS_ROLLOUT=${NGPUS_ROLLOUT:-8}

NNODES_TRAIN=${NNODES_TRAIN:-1}
NGPUS_TRAIN=${NGPUS_TRAIN:-8}

TEACHER_NNODES=${TEACHER_NNODES:-2}
TEACHER_NGPUS_PER_NODE=${TEACHER_NGPUS_PER_NODE:-8}
TEACHER_REPLICAS=${TEACHER_REPLICAS:-2}
TEACHER_TP=${TEACHER_TP:-8}

############################ Student Megatron (8 GPUs) ############################

ACTOR_TP=${ACTOR_TP:-2}
ACTOR_CP=${ACTOR_CP:-2}
ACTOR_EP=${ACTOR_EP:-2}
ACTOR_PP=${ACTOR_PP:-1}
ACTOR_ETP=${ACTOR_ETP:-1}

############################ Student Rollout (8 GPUs) ############################

INFER_TP=${INFER_TP:-4}
ROLLOUT_GPU_UTIL=${ROLLOUT_GPU_UTIL:-0.75}

############################ OPD Loss ############################

DISTILLATION_LOSS_MODE=${DISTILLATION_LOSS_MODE:-k1}
USE_POLICY_GRADIENT=${USE_POLICY_GRADIENT:-True}
USE_TASK_REWARDS=${USE_TASK_REWARDS:-False}

############################ Async / Throughput Knobs ############################

OPD_PHASE=${OPD_PHASE:-aggressive}

if [[ "${OPD_PHASE}" == "conservative" ]]; then
  GEN_BATCH=${GEN_BATCH:-1}
  N_RESP=${N_RESP:-2}
  STALENESS=${STALENESS:-0.0}
  TRIGGER_SYNC=${TRIGGER_SYNC:-1}
  PARTIAL_ROLLOUT=${PARTIAL_ROLLOUT:-False}
  REQUIRE_BATCHES=${REQUIRE_BATCHES:-1}
  TEACHER_MAX_BATCHED_TOKENS=${TEACHER_MAX_BATCHED_TOKENS:-4096}
  TEACHER_MAX_SEQS=${TEACHER_MAX_SEQS:-1}
else
  GEN_BATCH=${GEN_BATCH:-4}
  N_RESP=${N_RESP:-4}
  STALENESS=${STALENESS:-0.5}
  TRIGGER_SYNC=${TRIGGER_SYNC:-4}
  PARTIAL_ROLLOUT=${PARTIAL_ROLLOUT:-True}
  REQUIRE_BATCHES=${REQUIRE_BATCHES:-2}
  TEACHER_MAX_BATCHED_TOKENS=${TEACHER_MAX_BATCHED_TOKENS:-8192}
  TEACHER_MAX_SEQS=${TEACHER_MAX_SEQS:-2}
fi

MINI_BATCH=${MINI_BATCH:-64}
MICRO_BATCH=${MICRO_BATCH:-1}
TOTAL_ROLLOUT_STEPS=${TOTAL_ROLLOUT_STEPS:-200000}
TEST_FREQ=${TEST_FREQ:-20}
SAVE_FREQ=${SAVE_FREQ:-20}
LR=${LR:-5e-7}

export TENSORBOARD_DIR
export WANDB_PROJECT=${WANDB_PROJECT:-verl-opd-qwen3-32h100}
CKPT_ROOT=${CKPT_ROOT:-"/mnt/bn/search-nlp-vagcp/lvjiajun.a/RL/opd_runs/qwen3_30b_a3b_235b_moe_32h100"}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-"${CKPT_ROOT}/rollouts"}
TENSORBOARD_DIR=${TENSORBOARD_DIR:-"${CKPT_ROOT}/tensorboard"}
DEFAULT_LOCAL_DIR=${DEFAULT_LOCAL_DIR:-"${CKPT_ROOT}/checkpoints"}
TRAINER_LOGGER=${TRAINER_LOGGER:-'["console","tensorboard","wandb"]'}
RESUME_MODE=${RESUME_MODE:-auto}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-3}

OFFLOAD=${OFFLOAD:-True}
USE_MBRIDGE=${USE_MBRIDGE:-True}
USE_DIST_CKPT=${USE_DIST_CKPT:-False}

############################ Parameter Groups ############################

DATA=(
  data.train_files="['${TRAIN_FILE}']"
  data.val_files="['${TEST_FILE}']"
  data.prompt_key=prompt
  data.truncation=left
  data.max_prompt_length=${MAX_PROMPT}
  data.max_response_length=${MAX_RESPONSE}
  data.train_batch_size=0
  data.gen_batch_size=${GEN_BATCH}
  data.return_raw_chat=True
  data.filter_overlong_prompts=True
  data.shuffle=True
)

MODEL=(
  actor_rollout_ref.model.path="${STUDENT_MODEL}"
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.use_fused_kernels=False
  +actor_rollout_ref.model.override_config.model_config.max_position_embeddings=${MAX_LEN}
)

STUDENT=(
  actor_rollout_ref.hybrid_engine=False
  actor_rollout_ref.actor.strategy=megatron
  actor_rollout_ref.actor.use_rollout_log_probs=True
  actor_rollout_ref.actor.use_kl_loss=False
  actor_rollout_ref.actor.kl_loss_coef=0.0
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH}
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH}
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_LEN}
  actor_rollout_ref.actor.optim.lr=${LR}
  actor_rollout_ref.actor.optim.lr_warmup_steps=10
  actor_rollout_ref.actor.optim.lr_decay_style=constant
  actor_rollout_ref.actor.optim.weight_decay=0.1
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.actor.loss_agg_mode=token-mean
  actor_rollout_ref.actor.clip_ratio_low=0.2
  actor_rollout_ref.actor.clip_ratio_high=0.28
  actor_rollout_ref.actor.clip_ratio_c=10.0
  actor_rollout_ref.actor.megatron.use_mbridge=${USE_MBRIDGE}
  actor_rollout_ref.actor.megatron.use_dist_checkpointing=${USE_DIST_CKPT}
  actor_rollout_ref.actor.megatron.param_offload=${OFFLOAD}
  actor_rollout_ref.actor.megatron.grad_offload=${OFFLOAD}
  actor_rollout_ref.actor.megatron.optimizer_offload=${OFFLOAD}
  actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${ACTOR_TP}
  actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${ACTOR_PP}
  actor_rollout_ref.actor.megatron.context_parallel_size=${ACTOR_CP}
  actor_rollout_ref.actor.megatron.expert_model_parallel_size=${ACTOR_EP}
  actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ACTOR_ETP}
  +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True
  +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=True
  +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True
  +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True
  +actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs=True
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=flex
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
  +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
)

ROLLOUT=(
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.mode=async
  actor_rollout_ref.rollout.n=${N_RESP}
  actor_rollout_ref.rollout.calculate_log_probs=True
  actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_UTIL}
  actor_rollout_ref.rollout.tensor_model_parallel_size=${INFER_TP}
  actor_rollout_ref.rollout.max_model_len=${MAX_LEN}
  actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_LEN}
  actor_rollout_ref.rollout.enable_chunked_prefill=True
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.top_p=1.0
  actor_rollout_ref.rollout.top_k=-1
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH}
  actor_rollout_ref.rollout.enforce_eager=False
  actor_rollout_ref.rollout.free_cache_engine=True
  actor_rollout_ref.rollout.checkpoint_engine.backend=nccl
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1024
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0
  actor_rollout_ref.rollout.val_kwargs.top_p=0.7
  actor_rollout_ref.rollout.val_kwargs.do_sample=True
  actor_rollout_ref.rollout.val_kwargs.n=1
)

DISTILLATION=(
  distillation.enabled=True
  distillation.n_gpus_per_node=${TEACHER_NGPUS_PER_NODE}
  distillation.nnodes=${TEACHER_NNODES}
  distillation.teacher_models.teacher_model.model_path="${TEACHER_MODEL}"
  distillation.teacher_models.teacher_model.num_replicas=${TEACHER_REPLICAS}
  distillation.teacher_models.teacher_model.inference.name=vllm
  distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${TEACHER_TP}
  distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.88
  distillation.teacher_models.teacher_model.inference.max_model_len=${MAX_LEN}
  distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=${TEACHER_MAX_BATCHED_TOKENS}
  distillation.teacher_models.teacher_model.inference.max_num_seqs=${TEACHER_MAX_SEQS}
  distillation.teacher_models.teacher_model.inference.enforce_eager=False
  +distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm.max_logprobs=1
  distillation.distillation_loss.loss_mode=${DISTILLATION_LOSS_MODE}
  distillation.distillation_loss.use_policy_gradient=${USE_POLICY_GRADIENT}
  distillation.distillation_loss.use_task_rewards=${USE_TASK_REWARDS}
  distillation.distillation_loss.loss_max_clamp=10.0
  distillation.distillation_loss.log_prob_min_clamp=-10.0
)

ALGORITHM=(
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=False
  algorithm.kl_ctrl.kl_coef=0.0
  algorithm.rollout_correction.bypass_mode=True
)

REWARD=(
  reward.reward_manager.name=naive
)

TRAINER=(
  trainer.logger="${TRAINER_LOGGER}"
  trainer.project_name=verl-opd-qwen3-32h100
  trainer.experiment_name="30b-a3b_235b-moe_${OPD_PHASE}_32h100"
  trainer.default_local_dir="${DEFAULT_LOCAL_DIR}"
  trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}"
  trainer.val_before_train=True
  trainer.save_freq=${SAVE_FREQ}
  trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP}
  trainer.resume_mode=${RESUME_MODE}
  trainer.test_freq=${TEST_FREQ}
  trainer.log_val_generations=5
  trainer.nnodes=${NNODES_TRAIN}
  trainer.n_gpus_per_node=${NGPUS_TRAIN}
  +trainer.use_legacy_worker_impl=disable
)

ASYNC=(
  rollout.nnodes=${NNODES_ROLLOUT}
  rollout.n_gpus_per_node=${NGPUS_ROLLOUT}
  rollout.total_rollout_steps=${TOTAL_ROLLOUT_STEPS}
  async_training.staleness_threshold=${STALENESS}
  async_training.trigger_parameter_sync_step=${TRIGGER_SYNC}
  async_training.require_batches=${REQUIRE_BATCHES}
  async_training.partial_rollout=${PARTIAL_ROLLOUT}
  async_training.use_trainer_do_validate=False
)

RAY=()
if [[ -n "${RAY_ADDRESS}" ]]; then
  RAY+=(+ray_kwargs.ray_init.address="${RAY_ADDRESS}")
fi

if [[ -z "${VERL_TASK_RUNNER_NODE_ID:-}" ]]; then
  VERL_TASK_RUNNER_NODE_ID="$(
    "${PYTHON_BIN}" -c 'import os, ray
addr = os.environ.get("RAY_ADDRESS") or "auto"
ray.init(address=addr, ignore_reinit_error=True, logging_level="ERROR")
nodes = [n for n in ray.nodes() if n.get("Alive") and (n.get("Resources") or {}).get("GPU", 0) > 0]
if not nodes:
    raise SystemExit("no alive GPU Ray nodes found for TaskRunner pin")
print(nodes[0]["NodeID"])
ray.shutdown()' 2>/tmp/verl_task_runner_node.err
  )"
  export VERL_TASK_RUNNER_NODE_ID
fi

############################ Launch ############################

echo "============================================================"
echo "OPD fully_async 32xH100 | phase=${OPD_PHASE}"
echo "  Student: ${STUDENT_MODEL}"
echo "  Teacher: ${TEACHER_MODEL}"
echo "  Seq:     prompt=${MAX_PROMPT} response=${MAX_RESPONSE}"
echo "  GPUs:    rollout=${NGPUS_ROLLOUT} train=${NGPUS_TRAIN} teacher=$((TEACHER_NNODES * TEACHER_NGPUS_PER_NODE))"
echo "  Loss:    ${DISTILLATION_LOSS_MODE} (pg=${USE_POLICY_GRADIENT})"
echo "  Async:   staleness=${STALENESS} trigger=${TRIGGER_SYNC} partial=${PARTIAL_ROLLOUT}"
echo "  Save:    every ${SAVE_FREQ} param_version -> ${DEFAULT_LOCAL_DIR}"
echo "  Rollout: dump -> ${ROLLOUT_DATA_DIR}"
echo "  TB:      ${TENSORBOARD_DIR}"
echo "  Python:  ${PYTHON_BIN}"
echo "  Ray:     ${RAY_ADDRESS:-local/default}"
echo "  Driver:  ${VERL_TASK_RUNNER_NODE_ID}"
echo "============================================================"

"${PYTHON_BIN}" -m verl.experimental.fully_async_policy.fully_async_main \
  --config-path=config \
  --config-name=fully_async_ppo_megatron_trainer.yaml \
  critic.strategy=megatron \
  "${DATA[@]}" \
  "${MODEL[@]}" \
  "${STUDENT[@]}" \
  "${ROLLOUT[@]}" \
  "${DISTILLATION[@]}" \
  "${ALGORITHM[@]}" \
  "${REWARD[@]}" \
  "${TRAINER[@]}" \
  "${ASYNC[@]}" \
  "${RAY[@]}" \
  "$@"
