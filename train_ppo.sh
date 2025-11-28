set -e

export CUDA_VISIBLE_DEVICES=0
export DATA_DIR='data/debug_fake'

WAND_PROJECT='Search-R1-Debug'

export BASE_MODEL='Qwen/Qwen2.5-0.5B-Instruct'
export EXPERIMENT_NAME=debug-fake-retriever-ppo

# 如果不存在调试用数据，则自动生成极小规模样本，避免下载大数据集。
python - <<'PY'
import os
import pandas as pd

data_dir = "data/debug_fake"
os.makedirs(data_dir, exist_ok=True)

train_path = os.path.join(data_dir, "train.parquet")
test_path = os.path.join(data_dir, "test.parquet")

if not (os.path.exists(train_path) and os.path.exists(test_path)):
    rows = []
    for i in range(4):
        rows.append({
            "data_source": "toy",
            "prompt": [{"role": "user", "content": f"Answer the question: toy question {i}?"}],
            "ability": "fact-reasoning",
            "reward_model": {"style": "rule", "ground_truth": {"target": [f"toy answer {i}"]}},
            "extra_info": {"split": "train", "index": i},
        })
    pd.DataFrame(rows).to_parquet(train_path)
    pd.DataFrame(rows[:2]).to_parquet(test_path)
PY

export VLLM_ATTENTION_BACKEND=XFORMERS

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_data_num=4 \
    data.val_data_num=2 \
    data.train_batch_size=2 \
    data.val_batch_size=1 \
    data.max_prompt_length=1024 \
    data.max_response_length=128 \
    data.max_start_length=512 \
    data.max_obs_length=128 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    critic.optim.lr=5e-5 \
    critic.model.use_remove_padding=True \
    critic.optim.lr_warmup_steps_ratio=0.015 \
    critic.model.path=$BASE_MODEL \
    critic.model.enable_gradient_checkpointing=true \
    critic.ppo_micro_batch_size=1 \
    critic.model.fsdp_config.param_offload=true \
    critic.model.fsdp_config.grad_offload=true \
    critic.model.fsdp_config.optimizer_offload=true \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.no_think_rl=false \
    trainer.critic_warmup=0 \
    trainer.logger=[] \
    +trainer.val_only=false \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=1 \
    trainer.total_training_steps=2 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=2 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=1 \
    2>&1 | tee $EXPERIMENT_NAME.log
