#!/usr/bin/env sh

# 路径和模型配置
train_data_path='./configs/data.yaml'
# 请确保这里的路径指向你下载的、完整的教师模型 .pth 文件
teacher_pth_file="/root/autodl-tmp/neta-lumina-hf/consolidated.00-of-01.pth"
# 如果你想从一个已有的学生模型checkpoint继续训练，可以在这里指定路径
init_from_path=""

model=NextDiT_2B_GQA_patch2_Adaln_Refiner
precision=bf16

# 训练超参数
global_batch_size=1
micro_batch_size=1
lr=1e-5 # 学生模型 (生成器) 的学习率
critic_lr=2e-4 # 判别器/Critic的学习率

# 蒸馏策略配置
# 核心损失权重 (DMD + Original + GAN)
dmd_weight=1.0          # Distribution Matching Distillation 损失权重
original_weight=0.1     # 原始速度预测损失 (MSE) 权重
gan_weight=0.1          # 对抗性 GAN 损失权重 (设置为 0 来禁用 GAN)

# Two Time-scale Update Rule (TTUR)
critic_updates=5        # 每个生成器更新步骤中，Critic/判别器的更新次数

# 多步蒸馏配置 (通过反向模拟)
use_backward_simulation=true # 设置为 true 来启用反向模拟
num_distill_steps=4          # 蒸馏步数 (例如 1, 4, 8)

# 实验和日志
exp_name=${model}_distill_${num_distill_steps}steps_gan${gan_weight}_bs${global_batch_size}_lr${lr}
mkdir -p results/"$exp_name"

# 分布式训练设置 (单机单卡)
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="12345"
export RANK="0"
export WORLD_SIZE="1"
export LOCAL_RANK="0"
export LOCAL_WORLD_SIZE="1"

# 启动训练
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