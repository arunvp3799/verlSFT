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
aime24_avg10=$HOME/Repo/PartialRL/data/aime24_avg10.parquet
math_train=$HOME/Repo/PartialRL/data/math_train.parquet
math500=$HOME/Repo/PartialRL/data/math500.parquet
minerva_math_hardsplit=$HOME/Repo/PartialRL/data/minerva_math_hardsplit.parquet
minerva_math=$HOME/Repo/PartialRL/data/minerva_math.parquet
olympiad_bench=$HOME/Repo/PartialRL/data/olympiad_bench.parquet
olympiad_bench_hardsplit=$HOME/Repo/PartialRL/data/olympiad_bench_hardsplit.parquet
math500_hardsplit=$HOME/Repo/PartialRL/data/math500_hardsplit.parquet
aime24_avg10_hardsplit=$HOME/Repo/PartialRL/data/aime24_avg10_hardsplit.parquet
aime25_avg10=$HOME/Repo/PartialRL/data/aime25_avg10.parquet
gsm8k_test=$HOME/Repo/verl/data/gsm8k/test.parquet
aime_test=$HOME/Repo/verl/data/aime/aime_test.parquet
# val_files="['$aime24_avg10', '$math_train', '$math500', '$minerva_math_hardsplit', '$minerva_math', '$olympiad_bench', '$olympiad_bench_hardsplit', '$math500_hardsplit', '$aime24_avg10_hardsplit']"
# val_files="['$aime25_avg10','$aime24_avg10_hardsplit']"
val_files="['$gsm8k_test']"

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/Repo/verl/data/gsm8k/train.parquet \
    data.val_files="$val_files" \
    data.use_template=true \
    data.template_name=gsm8k \
    data.use_model_chat_template=false \
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