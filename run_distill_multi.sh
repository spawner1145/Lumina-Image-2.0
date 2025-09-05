#!/usr/bin/env sh

train_data_path='./configs/data.yaml'
teacher_pth_file="/root/autodl-tmp/neta-lumina-hf-simple/consolidated.00-of-01.pth"
init_from_path="" # 从头训练学生模型

model=NextDiT_2B_GQA_patch2_Adaln_Refiner
precision=bf16

# GPU 数量
NUM_GPUS=4 
# 设置每张卡的微批次大小 (Micro Batch Size per GPU)
# 如果显存有多，可以增加到2
MICRO_BATCH_SIZE_PER_GPU=1

# 全局批次大小 (会自动计算)
# 全局批次大小 = GPU数量 * 每张卡的微批次大小
global_batch_size=$((NUM_GPUS * MICRO_BATCH_SIZE_PER_GPU))

# 学习率 (根据全局批次大小进行线性缩放)
# 基础学习率 (例如，当全局批次为1时)
BASE_LR=1e-5
lr=$(echo "$BASE_LR * $global_batch_size" | bc -l)

dmd_weight=1.0
gan_weight=0.0
original_weight=0.1
critic_lr=2e-4
critic_updates=5

exp_name=${model}_distill_${NUM_GPUS}gpus_bs${global_batch_size}_lr${lr}
mkdir -p results/"$exp_name"

torchrun --nproc_per_node=$NUM_GPUS distill.py \
    --model ${model} \
    --data_path ${train_data_path} \
    --results_dir "results/${exp_name}" \
    --global_bsz_1024 ${global_batch_size} \
    --micro_bsz_1024 ${MICRO_BATCH_SIZE_PER_GPU} \
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
    2>&1 | tee -a "results/${exp_name}/output.log"