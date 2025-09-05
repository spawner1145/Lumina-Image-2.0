#!/usr/bin/env sh

train_data_path='./configs/data.yaml'
teacher_pth_file="/root/autodl-tmp/neta-lumina-hf/consolidated.00-of-01.pth"
init_from_path="" # 从头训练学生模型

model=NextDiT_2B_GQA_patch2_Adaln_Refiner
precision=bf16

global_batch_size=1
micro_batch_size=1
lr=1e-5

dmd_weight=1.0
gan_weight=0.0
original_weight=0.1
critic_lr=2e-4
critic_updates=5

# 多步蒸馏参数
use_backward_simulation=true # 设为 true 来启用反向模拟
num_distill_steps=4          # 设置蒸馏步数，例如 4 步

exp_name=${model}_distill_${num_distill_steps}steps_bs${global_batch_size}_lr${lr}
mkdir -p results/"$exp_name"

export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="12345"
export RANK="0"
export WORLD_SIZE="1"
export LOCAL_RANK="0"
export LOCAL_WORLD_SIZE="1"

python -u distill.py \
    --model ${model} \
    --data_path ${train_data_path} \
    --results_dir "results/${exp_name}" \
    --global_bsz_1024 ${global_batch_size} \
    --micro_bsz_1024 ${micro_batch_size} \
    --precision ${precision} \
    --grad_precision fp32 \
    --checkpointing \
    --lr ${lr} \
    --grad_clip 2.0 \
    --max_steps 300000 \
    --snr_type "lognorm" \
    --qk_norm \
    --teacher_ckpt_path ${teacher_pth_file} \
    --dmd_loss_weight ${dmd_weight} \
    --gan_loss_weight ${gan_weight} \
    --original_loss_weight ${original_weight} \
    --lr_critic ${critic_lr} \
    --critic_updates_per_step ${critic_updates} \
    --init_from "${init_from_path}" \
    --ckpt_every 1000 \
    --log_every 10 \
    --global_seed 2024 \
    --num_workers 4 \
    --cache_data_on_disk \
    --use_backward_simulation \
    --num_distill_steps ${num_distill_steps} \
    2>&1 | tee -a "results/${exp_name}/output.log"