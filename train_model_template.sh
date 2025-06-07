## Train a model with a inbuilt model template

set -x

if [ "$#" -lt 1 ]; then
    echo "Usage: run_qwen_05_sp2.sh <nproc_per_node> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
# save_path=$2

# Shift the arguments so $@ refers to the rest
shift 1

# Define Val Files
gsm8k_test=$HOME/Repo/verl/data/gsm8k/test.parquet
val_files="['$gsm8k_test']"

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/Repo/verl/data/gsm8k/train.parquet \
    data.val_files="$val_files" \
    data.use_template=false \
    data.use_model_chat_template=false \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.prompt_dict_keys=['question'] \
    data.response_dict_keys=['answer'] \
    data.micro_batch_size=8 \
    data.micro_batch_size_per_gpu=8 \
    data.train_batch_size=64 \
    data.max_length=6144 \
    model.partial_pretrain=Qwen/Qwen3-8B \
    model.use_liger=True \
    trainer.project_name=sft \
    trainer.experiment_name=gsm8k_qwen3_8b_sft \
    trainer.logger=['console'] \
    trainer.default_hdfs_dir=null $@ \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true \
    trainer.total_epochs=4 \
    trainer.default_local_dir=$HOME/Repo/verl/sft_models/gsm8k_qwen3_8b_sft_test