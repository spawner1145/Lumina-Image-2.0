"""
dmd2，同目录下有run_distill.sh
"""
import argparse
from collections import OrderedDict, defaultdict
import contextlib
from copy import deepcopy
from datetime import datetime
import functools
from functools import partial
import json
import logging
import os
import random
import socket
from time import time
import warnings
import torch.nn.functional as F

from PIL import Image
# import cairosvg
from diffusers import AutoencoderKL
import fairscale.nn.model_parallel.initialize as fs_init
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer

from data import DataNoReportException, ItemProcessor, MyDataset, read_general
from imgproc import generate_crop_size_list, to_rgb_if_rgba, var_center_crop
import models
from parallel import distributed_init, get_intra_node_process_group
from transport import create_transport
from util.misc import SmoothedValue

#############################################################################
#                               Data item Processor                             #
#############################################################################

class NonRGBError(DataNoReportException):
    pass

class T2IItemProcessor(ItemProcessor):
    def __init__(self, transform):
        self.image_transform = transform
        self.special_format_set = set()


    def process_item(self, data_item, training_mode=False):
        if "super_high_quality_caption" in data_item:
            url = data_item["image_path"]
            image = Image.open(read_general(url))
            text = data_item["super_high_quality_caption"]
            system_prompt = "You are an assistant designed to generate high-quality images with the highest degree of image-text alignment based on textual prompts. <Prompt Start> "  # noqa
        elif "image_path" in data_item and "prompt" in data_item:
            url = data_item["image_path"]
            image = Image.open(read_general(url))
            text = data_item["prompt"]
            system_prompt = "You are an assistant designed to generate high-quality images based on user prompts. <Prompt Start> "  # noqa
        elif "path" in data_item:
            url = data_item["path"]
            image = Image.open(read_general(url))
            text = data_item["prompt"]
            system_prompt = "You are an assistant designed to generate high-quality images based on user prompts. <Prompt Start> "  # noqa
        else:
            raise ValueError(f"Unrecognized item: {data_item}")

        if image.mode.upper() != "RGB":
            mode = image.mode.upper()
            if mode not in self.special_format_set:
                self.special_format_set.add(mode)
                print(mode, url)
            if mode == "RGBA":
                image = to_rgb_if_rgba(image)
            elif mode == "P" or mode == "L":
                image = image.convert("RGB")
            else:
                raise NonRGBError()

        image = self.image_transform(image)

        if text is None or text.strip() == "":
            text = ""
        text = system_prompt + text
        return image, text


#############################################################################
#                           Training Helper Functions                       #
#############################################################################

def apply_average_pool(latent, factor):
    """
    Apply average pooling to downsample the latent.
    """
    return F.avg_pool2d(latent, kernel_size=factor, stride=factor)

def dataloader_collate_fn(samples):
    image = [x[0] for x in samples]
    caps = [x[1] for x in samples]
    return image, caps


def get_train_sampler(dataset, rank, world_size, global_batch_size, max_steps, resume_step, seed):
    sample_indices = torch.empty([max_steps * global_batch_size // world_size], dtype=torch.long)
    epoch_id, fill_ptr, offs = 0, 0, 0
    while fill_ptr < sample_indices.size(0):
        g = torch.Generator()
        g.manual_seed(seed + epoch_id)
        epoch_sample_indices = torch.randperm(len(dataset), generator=g)
        epoch_id += 1
        epoch_sample_indices = epoch_sample_indices[(rank + offs) % world_size :: world_size]
        offs = (offs + world_size - len(dataset) % world_size) % world_size
        epoch_sample_indices = epoch_sample_indices[: sample_indices.size(0) - fill_ptr]
        sample_indices[fill_ptr : fill_ptr + epoch_sample_indices.size(0)] = epoch_sample_indices
        fill_ptr += epoch_sample_indices.size(0)
    return sample_indices[resume_step * global_batch_size // world_size :].tolist()


@torch.no_grad()
def update_ema(ema_model, model, decay=0.999): # Using a higher decay for EMA is common in distillation
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    assert set(ema_params.keys()) == set(model_params.keys())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def cleanup():
    dist.destroy_process_group()


def create_logger(logging_dir):
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ],
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def setup_lm_fsdp_sync(model: nn.Module) -> FSDP:
    model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in list(model.layers),
        ),
        process_group=get_intra_node_process_group(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(param_dtype=next(model.parameters()).dtype),
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    torch.cuda.synchronize()
    return model


def setup_fsdp_sync(model: nn.Module, args: argparse.Namespace) -> FSDP:
    model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in model.get_fsdp_wrap_module_list(),
        ),
        process_group=fs_init.get_data_parallel_group(),
        sharding_strategy={"fsdp": ShardingStrategy.FULL_SHARD, "sdp": ShardingStrategy.SHARD_GRAD_OP}[args.data_parallel],
        mixed_precision=MixedPrecision(
            param_dtype={"fp32": torch.float, "tf32": torch.float, "bf16": torch.bfloat16, "fp16": torch.float16}[args.precision],
            reduce_dtype={"fp32": torch.float, "tf32": torch.float, "bf16": torch.bfloat16, "fp16": torch.float16}[args.grad_precision or args.precision],
        ),
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    torch.cuda.synchronize()
    return model


def setup_mixed_precision(args):
    if args.precision == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif args.precision in ["bf16", "fp16", "fp32"]:
        pass
    else:
        raise NotImplementedError(f"Unknown precision: {args.precision}")


def encode_prompt(prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train=True):
    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        text_inputs = tokenizer(
            captions, padding=True, pad_to_multiple_of=8, max_length=256, truncation=True, return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids
        prompt_masks = text_inputs.attention_mask
        prompt_embeds = text_encoder(
            input_ids=text_input_ids.cuda(), attention_mask=prompt_masks.cuda(), output_hidden_states=True
        ).hidden_states[-2]
    return prompt_embeds, prompt_masks


#############################################################################
#                                Training Loop                              #
#############################################################################

def main(args):
    """
    Trains a new DiT model using distillation.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    distributed_init(args)
    dp_world_size = fs_init.get_data_parallel_world_size()
    dp_rank = fs_init.get_data_parallel_rank()
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    setup_mixed_precision(args)

    os.makedirs(args.results_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.results_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    if rank == 0:
        logger = create_logger(args.results_dir)
        tb_logger = SummaryWriter(os.path.join(args.results_dir, "tensorboard", datetime.now().strftime("%Y%m%d_%H%M%S_") + socket.gethostname()))
    else:
        logger = create_logger(None)
        tb_logger = None

    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    logger.info("Training arguments: " + json.dumps(args.__dict__, indent=2))

    logger.info(f"Setting-up language model: google/gemma-2-2b")
    tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/neta-lumina-hf/tokenizer", token=args.hf_token)
    tokenizer.padding_side = "right"
    text_encoder = AutoModel.from_pretrained("/root/autodl-tmp/neta-lumina-hf/text_encoder", torch_dtype=torch.bfloat16, token=args.hf_token).cuda()
    text_encoder = setup_lm_fsdp_sync(text_encoder)
    cap_feat_dim = text_encoder.config.hidden_size

    # Define the multi-step sampling schedule based on the paper
    # "Our 4-step model uses the following schedule: 999, 749, 499, 249, for a teacher model trained with 1000 steps."
    # We adapt this to a more general form.
    if args.use_backward_simulation:
        teacher_total_steps = 1000
        if args.num_distill_steps == 4:
            # Corresponds to paper's schedule for 4 steps
            timesteps = [999, 749, 499, 249]
        elif args.num_distill_steps == 1:
            timesteps = [999] # For single step, train at the beginning
        else:
            # A generic linear schedule for other step counts
            timesteps = np.linspace(teacher_total_steps - 1, 1, args.num_distill_steps, dtype=int).tolist()
        
        # Convert timesteps (e.g., 999) to continuous time t (e.g., 0.999)
        time_schedule = torch.tensor(timesteps, device=device) / teacher_total_steps
        logger.info(f"Using backward simulation with {args.num_distill_steps} steps. Time schedule: {time_schedule.tolist()}")

    # Model Setup (Student, Teacher, Critic)
    # Student Model
    logger.info(f"Creating student model: {args.model}")
    student_model = models.__dict__[args.model](in_channels=16, qk_norm=args.qk_norm, cap_feat_dim=cap_feat_dim)
    model_patch_size = student_model.patch_size

    # Teacher Model
    logger.info(f"Loading teacher model from: {args.teacher_ckpt_path}")
    teacher_model_ema = deepcopy(student_model)
    if dp_rank == 0:
        if not os.path.exists(args.teacher_ckpt_path):
            raise FileNotFoundError(f"Teacher checkpoint file not found at: {args.teacher_ckpt_path}")
        teacher_state_dict = torch.load(args.teacher_ckpt_path, map_location="cpu")
        new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in teacher_state_dict.items()}
        teacher_model_ema.load_state_dict(new_state_dict, strict=True)
    dist.barrier()
    teacher_model_ema = setup_fsdp_sync(teacher_model_ema, args)
    teacher_model_ema.eval()
    for param in teacher_model_ema.parameters():
        param.requires_grad = False
    logger.info("Teacher model loaded and frozen.")

    # Fake Critic Model
    logger.info(f"Creating fake critic model: {args.model}")
    fake_critic = deepcopy(student_model)
    student_model_ema = deepcopy(student_model)

    if args.checkpointing:
        checkpointing_list = list(student_model.get_checkpointing_wrap_module_list())
    else:
        checkpointing_list = []
    
    # FSDP Wrapping and EMA setup
    student_model = setup_fsdp_sync(student_model, args)
    student_model_ema = setup_fsdp_sync(student_model_ema, args)
    fake_critic = setup_fsdp_sync(fake_critic, args)

    if args.checkpointing:
        logger.info("Applying activation checkpointing...")
        non_reentrant_wrapper = partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        
        # 为每个模型生成自己的 checkpointing_list
        checkpointing_list_student = list(student_model.get_checkpointing_wrap_module_list())
        apply_activation_checkpointing(
            student_model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda submodule: submodule in checkpointing_list_student,
        )
        
        checkpointing_list_ema = list(student_model_ema.get_checkpointing_wrap_module_list())
        apply_activation_checkpointing(
            student_model_ema,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda submodule: submodule in checkpointing_list_ema,
        )
        
        checkpointing_list_critic = list(fake_critic.get_checkpointing_wrap_module_list())
        apply_activation_checkpointing(
            fake_critic,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda submodule: submodule in checkpointing_list_critic,
        )

    # Optimizer Setup
    opt = torch.optim.AdamW(student_model.parameters(), lr=args.lr, weight_decay=args.wd, eps=1e-15, betas=(0.9, 0.95))
    opt_critic = torch.optim.AdamW(fake_critic.parameters(), lr=args.lr_critic, weight_decay=args.wd, eps=1e-15, betas=(0.9, 0.95))
    
    resume_step = 0 # Resume logic can be added here if needed

    # Data Setup
    data_collection = {}
    train_res=1024
    logger.info(f"Creating data for resolution {train_res}")
    global_bsz = getattr(args, f"global_bsz_{train_res}")
    local_bsz = global_bsz // dp_world_size
    micro_bsz = getattr(args, f"micro_bsz_{train_res}")
    assert global_bsz % dp_world_size == 0, "Batch size must be divisible by data parallel world size."
    patch_size = 8 * model_patch_size
    max_num_patches = round((train_res / patch_size) ** 2)
    crop_size_list = generate_crop_size_list(max_num_patches, patch_size)
    image_transform = transforms.Compose([
        transforms.Lambda(functools.partial(var_center_crop, crop_size_list=crop_size_list, random_top_k=1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    dataset = MyDataset(args.data_path, item_processor=T2IItemProcessor(image_transform), cache_on_disk=args.cache_data_on_disk)
    sampler = get_train_sampler(dataset, dp_rank, dp_world_size, global_bsz, args.max_steps, resume_step, args.global_seed)
    loader = DataLoader(dataset, batch_size=local_bsz, sampler=sampler, num_workers=args.num_workers, pin_memory=True, collate_fn=dataloader_collate_fn)
    transport = create_transport("Linear", "velocity", None, None, None, snr_type=args.snr_type, do_shift=not args.no_shift, seq_len=(train_res // 16) ** 2)
    data_collection[train_res] = {"loader_iter": iter(loader), "local_bsz": local_bsz, "micro_bsz": micro_bsz, "transport": transport}

    vae = AutoencoderKL.from_pretrained("/root/autodl-tmp/neta-lumina-hf", subfolder="vae", torch_dtype=torch.bfloat16).to(device)

    # Training Loop
    logger.info(f"Starting distillation training for {args.max_steps:,} steps...")
    for step in range(resume_step, args.max_steps):
        data_pack = data_collection[train_res]
        
        x, caps = next(data_pack["loader_iter"])
        x = [img.to(device, non_blocking=True) for img in x]
        with torch.no_grad():
            vae_scale = 0.3611
            vae_shift = 0.1159
            x_latents = [(vae.encode(img[None].bfloat16()).latent_dist.mode()[0] - vae_shift) * vae_scale for img in x]
            cap_feats, cap_mask = encode_prompt(caps, text_encoder, tokenizer, args.caption_dropout_prob)

        # Student Model Update
        opt.zero_grad()
        
        loss_g_item = 0.0
        dmd_loss_item = 0.0
        original_loss_item = 0.0

        for mb_idx in range((data_pack["local_bsz"] - 1) // data_pack["micro_bsz"] + 1):
            mb_st = mb_idx * data_pack["micro_bsz"]
            mb_ed = min((mb_idx + 1) * data_pack["micro_bsz"], data_pack["local_bsz"])
            last_mb = mb_ed == data_pack["local_bsz"]

            x1_mb = x_latents[mb_st:mb_ed]
            cap_feats_mb = cap_feats[mb_st:mb_ed]
            cap_mask_mb = cap_mask[mb_st:mb_ed]
            model_kwargs = dict(cap_feats=cap_feats_mb, cap_mask=cap_mask_mb)

            # --- 修改开始：在这里添加 autocast 上下文 ---
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # --- 反向模拟核心逻辑 (保持不变) ---
                if args.use_backward_simulation:
                    with torch.no_grad():
                        current_step_index = random.randint(0, args.num_distill_steps - 1)
                        t = time_schedule[current_step_index]
                        t_broadcasted = t.repeat(len(x1_mb))
                        x0 = [torch.randn_like(x) for x in x1_mb]
                        if current_step_index == 0:
                            x_prev_step_denoised = x0
                        else:
                            x_sim = x0
                            for i in range(current_step_index):
                                t_sim = time_schedule[i].repeat(len(x_sim))
                                denoised_velocity = student_model_ema(x_sim, t_sim, **model_kwargs)
                                _, alpha_t_sim, sigma_t_sim = data_pack["transport"].path_sampler.plan(t_sim, x_sim, x_sim)
                                x_hat = [(xt_i - sigma_i * v_i) / alpha_i for xt_i, sigma_i, v_i, alpha_i in zip(x_sim, sigma_t_sim, denoised_velocity, alpha_t_sim)]
                                if i < current_step_index - 1:
                                    t_next = time_schedule[i+1].repeat(len(x_sim))
                                    _, alpha_next, sigma_next = data_pack["transport"].path_sampler.plan(t_next, x_sim, x_sim)
                                    noise_for_next_step = [torch.randn_like(x) for x in x_sim]
                                    x_sim = [(alpha_i * x_hat_i + sigma_i * noise_i) for alpha_i, x_hat_i, sigma_i, noise_i in zip(alpha_next, x_hat, sigma_next, noise_for_next_step)]
                            x_prev_step_denoised = x_hat
                        noise = [torch.randn_like(x) for x in x1_mb]
                        _, alpha_t, sigma_t = data_pack["transport"].path_sampler.plan(t_broadcasted, x_prev_step_denoised, x_prev_step_denoised)
                        xt = [(alpha_i * x_denoised_i + sigma_i * noise_i) for alpha_i, x_denoised_i, sigma_i, noise_i in zip(alpha_t, x_prev_step_denoised, sigma_t, noise)]
                        _, _, ut = data_pack["transport"].path_sampler.plan(t_broadcasted, x_prev_step_denoised, x_prev_step_denoised)
                else:
                    t, x0, x1 = data_pack["transport"].sample(x1_mb)
                    t_broadcasted = t
                    xt, ut = data_pack["transport"].path_sampler.plan(t, x0, x1)

                with torch.no_grad():
                    s_real = teacher_model_ema(xt, t_broadcasted, **model_kwargs)

                s_student = student_model(xt, t_broadcasted, **model_kwargs)

                dmd_loss = torch.stack([
                    F.mse_loss(s_student[i], s_real[i].detach()) for i in range(len(s_student))
                ]).mean()
                original_loss = torch.stack([
                    F.mse_loss(s_student[i], ut[i]) for i in range(len(s_student))
                ]).mean()
                
                gan_loss_g = torch.tensor(0.0, device=device)

                loss_g = (args.dmd_loss_weight * dmd_loss + args.original_loss_weight * original_loss + args.gan_loss_weight * gan_loss_g) / (data_pack["local_bsz"] / data_pack["micro_bsz"])
            
            loss_g.backward()
            loss_g_item += loss_g.item()
            dmd_loss_item += dmd_loss.item()
            original_loss_item += original_loss.item()

        student_model.clip_grad_norm_(max_norm=args.grad_clip)
        opt.step()
        update_ema(student_model_ema, student_model)

        # Critic Model Update (TTUR)
        for p in student_model.parameters(): p.requires_grad = False
        
        total_loss_c = 0.0
        for _ in range(args.critic_updates_per_step):
            opt_critic.zero_grad()
            critic_loss_item = 0.0
            for mb_idx in range((data_pack["local_bsz"] - 1) // data_pack["micro_bsz"] + 1):
                mb_st = mb_idx * data_pack["micro_bsz"]
                mb_ed = min((mb_idx + 1) * data_pack["micro_bsz"], data_pack["local_bsz"])

                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    with torch.no_grad():
                        # Re-use the same data from the student step for efficiency
                        x_mb = x_latents[mb_st:mb_ed]
                        cap_feats_mb = cap_feats[mb_st:mb_ed]
                        cap_mask_mb = cap_mask[mb_st:mb_ed]
                        t, x0, x1 = data_pack["transport"].sample(x_mb)
                        _, xt, _ = data_pack["transport"].path_sampler.plan(t, x0, x1)
                        model_kwargs = dict(cap_feats=cap_feats_mb, cap_mask=cap_mask_mb)
                        s_student_detached = student_model(xt, t, **model_kwargs)
                    
                    s_critic = fake_critic(xt, t, **model_kwargs)
    
                    loss_c = torch.stack([
                        F.mse_loss(s_critic[i], s_student_detached[i].detach()) for i in range(len(s_critic))
                    ]).mean()
                    
                    gan_loss_d = torch.tensor(0.0, device=device)
                    
                    loss_c_total = (loss_c + gan_loss_d) / (data_pack["local_bsz"] / data_pack["micro_bsz"])
                loss_c_total.backward()
                critic_loss_item += loss_c.item()

            fake_critic.clip_grad_norm_(max_norm=args.grad_clip)
            opt_critic.step()
            total_loss_c += critic_loss_item

        for p in student_model.parameters(): p.requires_grad = True

        # Logging
        if (step + 1) % args.log_every == 0 and rank == 0:
            avg_critic_loss = total_loss_c / args.critic_updates_per_step
            logger.info(f"Step {step+1:07d}: Student Loss: {loss_g_item:.4f}, Critic Loss: {avg_critic_loss:.4f}, DMD Loss: {dmd_loss_item:.4f}, Orig Loss: {original_loss_item:.4f}")
            if tb_logger:
                tb_logger.add_scalar("loss/student_total", loss_g_item, step)
                tb_logger.add_scalar("loss/critic_avg", avg_critic_loss, step)
                tb_logger.add_scalar("loss/dmd", dmd_loss_item, step)
                tb_logger.add_scalar("loss/original", original_loss_item, step)

        # Checkpointing
        if (step + 1) % args.ckpt_every == 0 or (step + 1) == args.max_steps:
            save_path = f"{checkpoint_dir}/{step + 1:07d}"
            os.makedirs(save_path, exist_ok=True)
            with FSDP.state_dict_type(student_model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(rank0_only=True, offload_to_cpu=True)):
                consolidated_model_state_dict = student_model.state_dict()
                if dp_rank == 0:
                    torch.save(consolidated_model_state_dict, os.path.join(save_path, f"consolidated.{fs_init.get_model_parallel_rank():02d}-of-{fs_init.get_model_parallel_world_size():02d}.pth"))
            dist.barrier()
            logger.info(f"Saved student model checkpoint to {save_path}")

    logger.info("Distillation training finished!")
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--cache_data_on_disk", default=False, action="store_true")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="NextDiT_2B_GQA_patch2_Adaln_Refiner")
    parser.add_argument("--max_steps", type=int, default=100_000, help="Number of training steps.")
    parser.add_argument("--global_bsz_1024", type=int, default=256)
    parser.add_argument("--micro_bsz_1024", type=int, default=1)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--ckpt_every", type=int, default=1000)
    parser.add_argument("--master_port", type=int, default=18181)
    parser.add_argument("--model_parallel_size", type=int, default=1)
    parser.add_argument("--data_parallel", type=str, choices=["sdp", "fsdp"], default="fsdp")
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--precision", choices=["fp32", "tf32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--grad_precision", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--no_auto_resume", action="store_false", dest="auto_resume")
    parser.add_argument("--resume", type=str, help="Resume training from a checkpoint folder.")
    parser.add_argument("--init_from", type=str, help="Initialize model weights from a checkpoint.")
    parser.add_argument("--grad_clip", type=float, default=2.0, help="Clip the L2 norm of the gradients.")
    parser.add_argument("--wd", type=float, default=0.0, help="Weight decay for the optimizer.")
    parser.add_argument("--qk_norm", action="store_true")
    parser.add_argument("--caption_dropout_prob", type=float, default=0.1)
    parser.add_argument("--snr_type", type=str, default="lognorm")
    parser.add_argument("--no_shift", action="store_true")
    parser.add_argument("--hf_token", type=str, default=None)

    parser.add_argument("--teacher_ckpt_path", type=str, required=True, help="Path to the single .pth file of the teacher model.")
    parser.add_argument("--dmd_loss_weight", type=float, default=1.0, help="Weight for the Distribution Matching Distillation loss.")
    parser.add_argument("--gan_loss_weight", type=float, default=0.0, help="Weight for the GAN generator loss (set to 0 to disable).")
    parser.add_argument("--original_loss_weight", type=float, default=0.1, help="Weight for the original velocity MSE loss.")
    parser.add_argument("--lr_critic", type=float, default=2e-4, help="Learning rate for the critic/discriminator optimizer.")
    parser.add_argument("--critic_updates_per_step", type=int, default=5, help="Number of critic updates per generator update (TTUR).")
    parser.add_argument("--use_backward_simulation", action="store_true", help="Enable backward simulation for multi-step distillation.")
    parser.add_argument("--num_distill_steps", type=int, default=1, help="Number of steps for the distilled student model.")
    
    args = parser.parse_args()

    main(args)
