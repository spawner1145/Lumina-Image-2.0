import argparse
import os
import builtins
import json
import math
import multiprocessing as mp
import random
import socket
import sys
import traceback
from types import SimpleNamespace

import gradio as gr
import numpy as np
from safetensors.torch import load_file
import torch
from torchvision.transforms.functional import to_pil_image

from transport import Sampler, create_transport
import models
from imgproc import generate_crop_size_list
from huggingface_hub import snapshot_download

try:
    from accelerate import cpu_offload
except ImportError:
    cpu_offload = None


class ModelFailure:
    """用于在进程间传递模型失败信息的类。"""
    def __init__(self, message="模型在后台发生未知错误。"):
        self.message = message
        print(f"模型失败: {message}")


def download_model_if_needed(model_id, base_cache_dir, hf_token=None):
    """
    检查本地是否存在模型，如果不存在则从Hugging Face Hub下载
    """
    clean_path = os.path.join(base_cache_dir, model_id.replace("/", "--"))

    try:
        print(f"正在检查本地是否存在模型 '{model_id}'...")
        snapshot_download(
            repo_id=model_id,
            local_dir=clean_path,
            local_files_only=True,
            local_dir_use_symlinks=False
        )
        print(f"成功！从本地离线加载模型: {clean_path}")

    except Exception:
        print(f"本地未找到模型 '{model_id}' 或模型文件不完整。")
        print(f"正在从 Hugging Face Hub 下载 '{model_id}'... (这可能需要一些时间)")

        try:
            snapshot_download(
                repo_id=model_id,
                local_dir=clean_path,
                token=hf_token,
                local_dir_use_symlinks=False
            )
            print(f"模型 '{model_id}' 下载完成。")
        except Exception as download_error:
            print(f"下载模型 '{model_id}' 失败: {download_error}")
            raise download_error

    return clean_path


def encode_prompt(prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train=True):
    """
    对输入的提示词进行编码
    """
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
            captions,
            padding=True,
            pad_to_multiple_of=8,
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )

        device = text_encoder.device if hasattr(text_encoder, 'device') else "cuda"
        text_input_ids = text_inputs.input_ids.to(device)
        prompt_masks = text_inputs.attention_mask.to(device)

        prompt_embeds = text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        ).hidden_states[-2]

    return prompt_embeds, prompt_masks

@torch.no_grad()
def model_main(request_queue, response_queue, mp_barrier):
    """
    运行在独立进程中的主模型循环，负责加载和推理
    """
    cache_dir = os.path.abspath('./cached_models')
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    from diffusers.models import AutoencoderKL
    from transformers import AutoModel, AutoTokenizer

    original_print = builtins.print
    def print_flush(*args, **kwargs):
        kwargs.setdefault("flush", True)
        original_print(*args, **kwargs)
    builtins.print = print_flush
    print(f"子进程启动。模型将缓存到目录: {cache_dir}")

    num_gpus = 1
    rank = 0
    master_port = find_free_port()
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(num_gpus)

    model_state = {
        "model": None, "vae": None, "text_encoder": None, "tokenizer": None,
        "loaded_ckpt_path": None, "loaded_precision": None, "loaded_hf_token": None,
        "offload_vae": False, "offload_text_encoder": False, "offload_dit": False,
    }

    mp_barrier.wait()

    while True:
        samples, z, cap_feats, cap_mask, model_kwargs = None, None, None, None, None
        try:
            settings_dict = request_queue.get()
            settings = SimpleNamespace(**settings_dict)
            metadata = settings_dict

            needs_reload = (
                model_state["model"] is None or
                model_state["loaded_ckpt_path"] != settings.ckpt or
                model_state["loaded_precision"] != settings.precision or
                model_state["loaded_hf_token"] != settings.hf_token or
                model_state["offload_vae"] != settings.offload_vae or
                model_state["offload_text_encoder"] != settings.offload_text_encoder or
                model_state["offload_dit"] != settings.offload_dit
            )

            if needs_reload:
                print("检测到模型或Offload设置变更，正在重新加载模型...")
                for k in ["model", "vae", "text_encoder", "tokenizer"]:
                    if model_state[k] is not None:
                        del model_state[k]
                model_state.update({k: None for k in model_state})
                torch.cuda.empty_cache()

                if any([settings.offload_vae, settings.offload_text_encoder, settings.offload_dit]):
                    if cpu_offload is None:
                        raise ImportError("错误: 您选择启用CPU Offload，但 `accelerate` 库未安装。请运行 `pip install accelerate`。")

                ckpt_path = settings.ckpt
                ckpt_dir = os.path.dirname(ckpt_path)
                train_args_path = os.path.join(ckpt_dir, "model_args.pth")
                if not os.path.exists(train_args_path):
                    raise FileNotFoundError(f"错误: 在模型目录 {ckpt_dir} 中找不到 'model_args.pth'。")

                train_args = torch.load(train_args_path, weights_only=False)
                print("加载的模型参数:", json.dumps(train_args.__dict__, indent=2))
                dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[settings.precision]

                print("正在准备依赖模型 (离线优先)...")
                vae_model_id = "black-forest-labs/FLUX.1-dev"
                text_encoder_model_id = "google/gemma-2-2b"
                vae_local_path = download_model_if_needed(vae_model_id, cache_dir, settings.hf_token)
                text_encoder_local_path = download_model_if_needed(text_encoder_model_id, cache_dir, settings.hf_token)

                print(f"正在加载 VAE (Offload: {settings.offload_vae})...")
                vae_device_map = "auto" if settings.offload_vae else "cuda"
                model_state["vae"] = AutoencoderKL.from_pretrained(
                    vae_local_path, subfolder="vae", token=settings.hf_token,
                    device_map=vae_device_map
                )
                if vae_device_map != "auto":
                    model_state["vae"] = model_state["vae"].cuda()

                print(f"正在加载文本编码器 (Offload: {settings.offload_text_encoder})...")
                encoder_device_map = "auto" if settings.offload_text_encoder else "cuda"
                model_state["text_encoder"] = AutoModel.from_pretrained(
                    text_encoder_local_path, torch_dtype=dtype, device_map=encoder_device_map, token=settings.hf_token
                ).eval()

                model_state["tokenizer"] = AutoTokenizer.from_pretrained(
                    text_encoder_local_path, token=settings.hf_token
                )
                model_state["tokenizer"].padding_side = "right"

                cap_feat_dim = model_state["text_encoder"].config.hidden_size

                print(f"正在创建 DiT 模型: {train_args.model} (Offload: {settings.offload_dit})")
                model_state["model"] = models.__dict__[train_args.model](
                    in_channels=16, qk_norm=train_args.qk_norm, cap_feat_dim=cap_feat_dim,
                )

                print(f"正在从 '{ckpt_path}' 加载模型权重...")
                map_location = "cpu" if settings.offload_dit else "cuda"
                if ckpt_path.endswith('.safetensors'):
                    state_dict = load_file(ckpt_path, device=map_location)
                else:
                    state_dict = torch.load(ckpt_path, map_location=map_location, weights_only=False)

                model_state["model"].load_state_dict(state_dict, strict=True)

                if settings.offload_dit:
                    print("正在为 DiT 模型启用 CPU Offload...")
                    model_state["model"].eval()
                    cpu_offload(model_state["model"], execution_device="cuda")
                else:
                    model_state["model"].eval().to("cuda", dtype=dtype)
                
                model_state.update({
                    "loaded_ckpt_path": settings.ckpt, "loaded_precision": settings.precision,
                    "loaded_hf_token": settings.hf_token, "offload_vae": settings.offload_vae,
                    "offload_text_encoder": settings.offload_text_encoder, "offload_dit": settings.offload_dit
                })
                print("模型加载完成！")
            else:
                print("模型设置未变更，将使用已缓存的模型。")

            model = model_state["model"]
            vae = model_state["vae"]
            text_encoder = model_state["text_encoder"]
            tokenizer = model_state["tokenizer"]
            dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[settings.precision]
            
            with torch.no_grad(), torch.autocast("cuda", dtype):
                system_prompt = settings.system_type
                cap = system_prompt + settings.cap
                neg_cap = system_prompt + settings.neg_cap if settings.neg_cap else ""
                print("接收到生成任务，参数:", json.dumps(metadata, indent=2))
                
                if settings.solver == "dpm":
                    transport = create_transport("Linear", "velocity")
                    sampler = Sampler(transport)
                    sample_fn = sampler.sample_dpm(model.forward_with_cfg)
                else:
                    transport = create_transport(
                        settings.path_type, settings.prediction, settings.loss_weight,
                        settings.train_eps, settings.sample_eps,
                    )
                    sampler = Sampler(transport)
                    sample_fn = sampler.sample_ode(
                        sampling_method=settings.solver, num_steps=settings.num_sampling_steps,
                        atol=settings.atol, rtol=settings.rtol, reverse=settings.reverse,
                        time_shifting_factor=settings.t_shift,
                    )

                latent_w, latent_h = settings.width // 8, settings.height // 8

                ui_seed = int(settings.seed)
                if ui_seed == -1:
                    actual_seed = random.randint(0, 2**32 - 1)
                    print(f"UI 种子为 -1，已生成随机种子: {actual_seed}")
                else:
                    actual_seed = ui_seed
                torch.random.manual_seed(actual_seed)
                metadata["seed"] = actual_seed

                z = torch.randn([1, 16, latent_h, latent_w], device="cuda").to(dtype)
                z = z.repeat(2, 1, 1, 1) 

                prompts_to_encode = [cap] + ([neg_cap] if neg_cap else [""])
                cap_feats, cap_mask = encode_prompt(prompts_to_encode, text_encoder, tokenizer, 0.0)
                cap_mask = cap_mask.to(cap_feats.device)

                model_kwargs = dict(
                    cap_feats=cap_feats, cap_mask=cap_mask,
                    cfg_scale=settings.cfg_scale, cfg_trunc=settings.cfg_trunc,
                    renorm_cfg=(True if settings.renorm_cfg == 'True' else (False if settings.renorm_cfg == 'False' else float(settings.renorm_cfg))),
                )

                print(f"开始采样... Steps: {settings.num_sampling_steps}, CFG: {settings.cfg_scale}, Seed: {actual_seed}")
                if settings.solver == "dpm":
                    samples = sample_fn(
                        z, steps=settings.num_sampling_steps, order=2,
                        skip_type="time_uniform_flow", method="multistep",
                        flow_shift=settings.t_shift, model_kwargs=model_kwargs
                    )
                else:
                    samples = sample_fn(z, model.forward_with_cfg, **model_kwargs)[-1]
                
                samples = samples[:1]

                vae_scale = 0.3611
                vae_shift = 0.1159
                vae_input_device = vae.device if hasattr(vae, 'device') else "cuda"
                samples = vae.decode(samples.to(vae_input_device) / vae_scale + vae_shift).sample
                
                samples = (samples + 1.0) / 2.0
                samples.clamp_(0.0, 1.0)
                img = to_pil_image(samples[0, :].cpu().float())
                
                print("图像生成完成！")
                response_queue.put((img, metadata))

        except Exception as e:
            print(traceback.format_exc())
            response_queue.put(ModelFailure(traceback.format_exc()))
        
        finally:
            print("正在删除中间变量并清理CUDA缓存...")
            del samples, z, cap_feats, cap_mask, model_kwargs
            torch.cuda.empty_cache()


def find_free_port() -> int:
    """寻找一个可用的端口。"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def find_ckpt_files(ckpt_dir="./ckpt"):
    """在指定目录中寻找支持的模型文件。"""
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
        print(f"'{ckpt_dir}' 目录不存在，已自动创建。请将模型文件放入此目录中。")
        return []

    supported_extensions = ['.safetensors', '.pth']
    ckpt_files = []
    for root, _, files in os.walk(ckpt_dir):
        for file in files:
            if file.lower() == 'model_args.pth':
                continue
            
            if any(file.lower().endswith(ext) for ext in supported_extensions):
                ckpt_files.append(os.path.join(root, file))
    return ckpt_files

def none_or_str(value):
    """处理Gradio Dropdown可能返回 "None" 字符串的情况。"""
    return None if value == "None" else value

def main():
    """主函数，负责启动多进程和Gradio UI。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860, help="Web UI 运行的端口。")
    args = parser.parse_args()

    if sys.platform != "win32":
        try:
            mp.set_start_method("fork", force=True)
            print("多进程启动方式设置为: fork")
        except RuntimeError:
            print("警告: 'fork' 启动方法已设置或不可用。")
    else:
        print("检测到 Windows 系统，使用默认的 'spawn' 启动方式。")

    num_gpus = 1
    request_queue = mp.Queue()
    response_queue = mp.Queue()
    mp_barrier = mp.Barrier(num_gpus + 1)

    # 启动模型子进程
    p = mp.Process(
        target=model_main,
        args=(request_queue, response_queue, mp_barrier),
        daemon=True
    )
    p.start()

    ckpt_list = find_ckpt_files()
    if not ckpt_list:
        print("严重错误：在 './ckpt' 目录中没有找到任何模型文件。UI 将无法正常工作。")
    
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Neta Lumina WebUI")

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Accordion("1. 模型设置", open=True):
                    ckpt = gr.Dropdown(choices=ckpt_list, value=ckpt_list[0] if ckpt_list else "无模型文件,请添加后重启", label="模型文件 (Checkpoint)", interactive=bool(ckpt_list))
                    precision = gr.Dropdown(["bf16", "fp16", "fp32"], value="bf16", label="运行精度 (Precision)")
                    hf_token = gr.Textbox(label="Hugging Face Token", placeholder="用于访问私有模型的HF Token", type="password")

                with gr.Accordion("2. 生成核心参数", open=True):
                    cap = gr.Textbox(lines=3, label="正向提示 (Prompt)", value="A majestic lion overlooking the savannah at sunset, photorealistic, 8k")
                    neg_cap = gr.Textbox(lines=2, label="反向提示 (Negative Prompt)", value="blurry, low quality, cartoon, watermark, text")
                    system_type = gr.Dropdown(choices=["You are an assistant designed to generate high-quality images with the highest degree of image-text alignment based on textual prompts.", ""], value="You are an assistant designed to generate high-quality images with the highest degree of image-text alignment based on textual prompts.", label="系统提示类型", max_choices=1)
                    with gr.Row():
                        width = gr.Slider(256, 2048, value=1024, step=64, label="宽度 (Width)")
                        height = gr.Slider(256, 2048, value=1024, step=64, label="高度 (Height)")
                    with gr.Row():
                        num_sampling_steps = gr.Slider(1, 100, value=20, step=1, label="采样步数")
                        seed = gr.Slider(-1, 100000, value=-1, step=1, label="种子 (-1 代表随机)")
                    with gr.Row():
                        cfg_scale = gr.Slider(1.0, 20.0, value=4.0, step=0.5, label="CFG Scale")
                        cfg_trunc = gr.Slider(0, 1, value=0.25, step=0.01, label="CFG Truncation")

                with gr.Accordion("3. 显存优化 (CPU Offload)", open=False):
                    gr.Markdown("勾选以将对应模型的部分或全部移至CPU内存，可节省显存但会降低速度。**需要 `accelerate` 库。**")
                    offload_text_encoder = gr.Checkbox(label="卸载文本编码器 (Text Encoder)", value=True)
                    offload_vae = gr.Checkbox(label="卸载 VAE", value=False)
                    offload_dit = gr.Checkbox(label="卸载主模型 (DiT)", value=False)

                with gr.Accordion("4. 采样器设置 (高级)", open=False):
                    solver = gr.Dropdown(["euler", "midpoint", "rk4", "dpm"], value="midpoint", label="采样器 (Solver)")
                    t_shift = gr.Slider(1, 20, value=6, step=1, label="时间步移 (Time Shift)")
                    renorm_cfg = gr.Dropdown(["True", "False", "2.0"], value="True", label="CFG Renormalization")

                with gr.Accordion("5. Transport & ODE 设置 (专家)", open=False):
                    path_type = gr.Dropdown(["Linear", "GVP", "VP"], value="Linear", label="Path Type")
                    prediction = gr.Dropdown(["velocity", "score", "noise"], value="velocity", label="Prediction")
                    loss_weight = gr.Dropdown(["None", "velocity", "likelihood"], value="None", label="Loss Weight")
                    atol = gr.Number(value=1e-6, label="ODE Absolute Tolerance")
                    rtol = gr.Number(value=1e-3, label="ODE Relative Tolerance")
                    reverse = gr.Checkbox(value=False, label="Reverse ODE Solver")
                    sample_eps = gr.Number(value=None, label="Sample Epsilon (可选)")
                    train_eps = gr.Number(value=None, label="Train Epsilon (可选)")

                submit_btn = gr.Button("生成图像 (Generate)", variant="primary")

            with gr.Column(scale=1):
                output_img = gr.Image(label="生成结果", interactive=False, show_download_button=True)
                with gr.Accordion("生成参数详情", open=True):
                    gr_metadata = gr.JSON(label="Metadata")

        gr.Examples(
            [
                ["A charcoal sketch of Istanbul with the iconic Hagia Sophia Mosque. The city streets wind through the landscape, with bright houses, trees, and flowers. There are puddles with raindrops and reflections of light and shadow. The Hagia Sophia stands tall and is crafted with precision. The background contains mountains and a bridge. The artist's signature 'brewozxy7' and the date 'October 2024' are in the lower left corner."],
                ["一个剑客，武侠风，红色腰带，戴着斗笠，低头，盖住眼睛，白色背景，细致，精品，杰作，水墨画，墨烟，墨云，泼墨，色带，墨水，墨黑白莲花，光影艺术，笔触。"],
                ["Aesthetic photograph of a bouquet of pink and white ranunculus flowers in a clear glass vase, centrally positioned on a wooden surface. The flowers are in full bloom, displaying intricate layers of petals with a soft gradient from pale pink to white. The vase is filled with water, visible through the clear glass, and the stems are submerged. Photorealistic, shallow depth of field, soft natural lighting, warm color palette, high contrast, glossy texture, tranquil, visually balanced."]
            ],
            inputs=[cap]
        )

        all_inputs = [
            ckpt, precision, hf_token, cap, neg_cap, system_type, width, height,
            num_sampling_steps, seed, cfg_scale, cfg_trunc,
            offload_text_encoder, offload_vae, offload_dit,
            solver, t_shift, renorm_cfg, path_type, prediction, loss_weight,
            atol, rtol, reverse, sample_eps, train_eps
        ]
        
        input_names = [
            "ckpt", "precision", "hf_token", "cap", "neg_cap", "system_type", "width", "height",
            "num_sampling_steps", "seed", "cfg_scale", "cfg_trunc",
            "offload_text_encoder", "offload_vae", "offload_dit",
            "solver", "t_shift", "renorm_cfg", "path_type", "prediction", "loss_weight",
            "atol", "rtol", "reverse", "sample_eps", "train_eps"
        ]

        def on_submit(*args):
            if not ckpt_list:
                raise gr.Error("错误：'./ckpt' 目录中没有模型文件。请添加模型文件并重启程序。")

            settings = dict(zip(input_names, args))
            settings['loss_weight'] = none_or_str(settings['loss_weight'])

            request_queue.put(settings)
            
            gr.Info("任务已提交，正在后台生成图像...")
            result = response_queue.get()

            if isinstance(result, ModelFailure):
                raise gr.Error(f"模型生成失败！\n错误详情: {result.message}")
            
            img, metadata = result
            return img, metadata

        submit_btn.click(on_submit, all_inputs, [output_img, gr_metadata])

    mp_barrier.wait()
    print(f"Gradio UI 已就绪, 请访问: http://127.0.0.1:{args.port} 或对应的公网地址")
    demo.queue().launch(server_name="0.0.0.0", server_port=args.port, share=True)


if __name__ == "__main__":
    main()
