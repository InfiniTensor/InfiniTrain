#!/usr/bin/env python3

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoConfig, AutoModel, AutoTokenizer


MAGIC = b"FM9MM001"
VPM_MAGIC = b"FM9VP001"
VPM_INPUT_MAGIC = b"FM9VI001"


class FP32MasterAdamW:
    def __init__(self, named_parameters, learning_rate, beta2, weight_decay):
        self.model_parameters = [parameter for _name, parameter in named_parameters]
        self.master_parameters = [
            parameter.detach().float().clone().requires_grad_(True) for parameter in self.model_parameters
        ]
        decay = [master for master, model in zip(self.master_parameters, self.model_parameters) if model.ndim > 1]
        no_decay = [master for master, model in zip(self.master_parameters, self.model_parameters) if model.ndim <= 1]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=learning_rate,
            betas=(0.9, beta2),
            eps=1e-8,
        )

    def zero_grad(self):
        for parameter in self.model_parameters:
            parameter.grad = None
        self.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step(self):
        for model, master in zip(self.model_parameters, self.master_parameters):
            master.grad = None if model.grad is None else model.grad.detach().float()
        self.optimizer.step()
        for model, master in zip(self.model_parameters, self.master_parameters):
            model.copy_(master.to(dtype=model.dtype))


def build_adamw(named_parameters, learning_rate, beta2, weight_decay, fp32_master_weights):
    if fp32_master_weights:
        return FP32MasterAdamW(named_parameters, learning_rate, beta2, weight_decay)
    decay = [parameter for _name, parameter in named_parameters if parameter.ndim > 1]
    no_decay = [parameter for _name, parameter in named_parameters if parameter.ndim <= 1]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, beta2),
        eps=1e-8,
    )


def build_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


def move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


def write_sample(path, input_ids, labels, image_indices, vision_embeddings):
    input_ids = input_ids.detach().cpu().to(torch.int64).contiguous().numpy()
    labels = labels.detach().cpu().to(torch.int64).contiguous().numpy()
    image_indices = image_indices.detach().cpu().to(torch.int64).contiguous().numpy()
    vision_embeddings = vision_embeddings.detach().cpu().to(torch.float32).contiguous().numpy()
    batch, sequence = input_ids.shape
    visual_tokens, hidden = vision_embeddings.shape
    vision_embeddings = vision_embeddings.reshape(batch, visual_tokens, hidden)
    image_indices = image_indices.reshape(batch, visual_tokens)
    with path.open("wb") as handle:
        handle.write(struct.pack("<8sqqqq", MAGIC, batch, sequence, visual_tokens, hidden))
        handle.write(input_ids.tobytes(order="C"))
        handle.write(labels.tobytes(order="C"))
        handle.write(image_indices.tobytes(order="C"))
        handle.write(vision_embeddings.tobytes(order="C"))


def write_vpm_sample(path, features, target_size):
    features = features.detach().cpu().to(torch.float32).contiguous().numpy()
    batch, tokens, hidden = features.shape
    height, width = (int(target_size[0]), int(target_size[1]))
    if tokens < height * width:
        raise RuntimeError(f"VPM output has {tokens} tokens but target size requires {height * width}")
    with path.open("wb") as handle:
        handle.write(struct.pack("<8sqqqqq", VPM_MAGIC, batch, tokens, height, width, hidden))
        handle.write(features.tobytes(order="C"))


def write_vpm_input(path, pixel_values_list, target_sizes):
    all_pixel_values = []
    for pixel_values in pixel_values_list:
        all_pixel_values.extend([item.flatten(end_dim=1).permute(1, 0) for item in pixel_values])
    if not all_pixel_values:
        raise RuntimeError("the selected sample has no VPM pixel input")
    pixels = torch.nn.utils.rnn.pad_sequence(all_pixel_values, batch_first=True, padding_value=0.0)
    batch, width, channels_times_height = pixels.shape
    if channels_times_height % 3 != 0:
        raise RuntimeError("VPM pixel input does not have three channels")
    pixels = pixels.permute(0, 2, 1).reshape(batch, 3, channels_times_height // 3, width)
    sizes = [item for item in target_sizes if isinstance(item, torch.Tensor)] if isinstance(target_sizes, list) else target_sizes
    sizes = torch.vstack(sizes) if isinstance(sizes, list) else sizes
    if sizes.shape[0] != batch or not torch.all(sizes == sizes[0]):
        raise RuntimeError("VPM input export currently requires a uniform target size")
    pixels = pixels.detach().cpu().float().contiguous().numpy()
    height_patches, width_patches = (int(sizes[0, 0]), int(sizes[0, 1]))
    with path.open("wb") as handle:
        handle.write(struct.pack(
            "<8sqqqqqq",
            VPM_INPUT_MAGIC,
            batch,
            pixels.shape[1],
            pixels.shape[2],
            pixels.shape[3],
            height_patches,
            width_patches,
        ))
        handle.write(pixels.tobytes(order="C"))


def main():
    parser = argparse.ArgumentParser(
        description="Export one official FM9G4B-V sample at the Resampler/LLM boundary for InfiniTrain alignment."
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--finetune-dir", type=Path, required=True)
    parser.add_argument("--data-json", type=Path, required=True)
    parser.add_argument("--output-bin", type=Path, required=True)
    parser.add_argument("--reference-grad-bin", type=Path, required=True)
    parser.add_argument("--reference-json", type=Path, required=True)
    parser.add_argument("--vpm-output-bin", type=Path)
    parser.add_argument("--vpm-input-bin", type=Path)
    parser.add_argument("--vpm-input-only", action="store_true")
    parser.add_argument("--vpm-reference-grad-bin", type=Path)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--attention", choices=("eager", "flash_attention_2"), default="eager")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-slice-nums", type=int, default=1)
    parser.add_argument("--parameter-grad-name")
    parser.add_argument("--parameter-grad-bin", type=Path)
    parser.add_argument("--parameter-output-name")
    parser.add_argument("--parameter-output-bin", type=Path)
    parser.add_argument("--parameter-initial-output-name")
    parser.add_argument("--parameter-initial-output-bin", type=Path)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--warmup-iterations", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.0)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--fp32-master-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tune-vision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tune-llm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.iterations <= 0 or not 0 <= args.warmup_iterations < args.iterations:
        parser.error("require iterations > 0 and 0 <= warmup-iterations < iterations")
    if args.learning_rate < 0 or args.max_grad_norm < 0:
        parser.error("learning-rate and max-grad-norm must be non-negative")
    if bool(args.parameter_grad_name) != bool(args.parameter_grad_bin):
        parser.error("--parameter-grad-name and --parameter-grad-bin must be used together")
    if bool(args.parameter_output_name) != bool(args.parameter_output_bin):
        parser.error("--parameter-output-name and --parameter-output-bin must be used together")
    if bool(args.parameter_initial_output_name) != bool(args.parameter_initial_output_bin):
        parser.error("--parameter-initial-output-name and --parameter-initial-output-bin must be used together")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sys.path.insert(0, str(args.finetune_dir.resolve()))
    from dataset import SupervisedDataset, data_collator

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    if args.attention == "eager":
        config._attn_implementation = "eager"
        config._attn_implementation_internal = "eager"
        config.vision_config._attn_implementation = "eager"
    config.slice_config.max_slice_nums = args.max_slice_nums
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    raw_data = json.loads(args.data_json.read_text(encoding="utf-8"))
    for item in raw_data:
        image = item.get("image")
        if isinstance(image, str) and not Path(image).is_absolute():
            item["image"] = str((args.data_json.parent / image).resolve())
        elif isinstance(image, dict):
            item["image"] = {
                key: str((args.data_json.parent / value).resolve()) if not Path(value).is_absolute() else value
                for key, value in image.items()
            }
    dataset = SupervisedDataset(
        raw_data,
        build_transform(),
        tokenizer,
        slice_config=config.slice_config.to_dict(),
        patch_size=config.patch_size,
        query_nums=config.query_num,
        batch_vision=config.batch_vision_input,
        max_length=args.max_length,
    )
    batch = data_collator([dataset[0]], max_length=args.max_length)
    if args.vpm_input_bin:
        args.vpm_input_bin.parent.mkdir(parents=True, exist_ok=True)
        write_vpm_input(args.vpm_input_bin, batch["pixel_values"], batch["tgt_sizes"])
    if args.vpm_input_only:
        if not args.vpm_input_bin:
            raise RuntimeError("--vpm-input-only requires --vpm-input-bin")
        print(f"exported VPM input: {args.vpm_input_bin}")
        return

    model = AutoModel.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        torch_dtype=dtype,
        config=config,
        attn_implementation=args.attention,
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.train()
    model.llm.requires_grad_(args.tune_llm)
    model.vpm.requires_grad_(args.tune_vision)
    model.resampler.requires_grad_(args.tune_vision)
    labels = batch.pop("labels").to("cuda", dtype=torch.int64)
    batch = move_to_device(batch, "cuda")

    named_trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not named_trainable:
        raise RuntimeError("no trainable parameters selected")
    optimizer = None
    if args.learning_rate > 0:
        optimizer = build_adamw(
            named_trainable,
            args.learning_rate,
            args.adam_beta2,
            args.weight_decay,
            args.fp32_master_weights and dtype != torch.float32,
        )
    if args.parameter_initial_output_name:
        parameter = dict(model.named_parameters()).get(args.parameter_initial_output_name)
        if parameter is None:
            raise RuntimeError(f"unknown initial parameter: {args.parameter_initial_output_name}")
        args.parameter_initial_output_bin.parent.mkdir(parents=True, exist_ok=True)
        parameter.detach().cpu().float().contiguous().numpy().tofile(args.parameter_initial_output_bin)

    captured_vpm = {}

    def capture_vpm_output(_module, _inputs, output):
        hidden_state = output.last_hidden_state
        hidden_state.retain_grad()
        captured_vpm["hidden_state"] = hidden_state

    hook = model.vpm.register_forward_hook(capture_vpm_output) if args.vpm_output_bin else None
    losses = []
    elapsed_ms_per_step = []
    grad_norms = []
    clip_scales = []
    peak_allocated_mb_per_step = []
    peak_reserved_mb_per_step = []
    parameter_grad = None
    for step in range(args.iterations):
        if optimizer is not None:
            optimizer.zero_grad()
        else:
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()

        inputs_embeds, vision_hidden_states = model.get_vllm_embedding(batch)
        vision = vision_hidden_states[0]
        if not isinstance(vision, torch.Tensor) or vision.numel() == 0:
            raise RuntimeError("the selected sample has no visual embeddings")
        vision.retain_grad()
        image_indices = torch.cat(
            [torch.arange(int(start), int(end), device="cuda") for start, end in batch["image_bound"][0]], dim=0
        )
        flat_vision = vision.reshape(-1, vision.shape[-1])
        if image_indices.numel() != flat_vision.shape[0]:
            raise RuntimeError(
                f"image token count {image_indices.numel()} does not match visual token count {flat_vision.shape[0]}"
            )

        outputs = model.llm(
            input_ids=None,
            position_ids=batch["position_ids"].to(torch.int64),
            inputs_embeds=inputs_embeds,
            use_cache=False,
        )
        logits = outputs.logits.reshape(-1, outputs.logits.shape[-1])
        loss = F.cross_entropy(logits, labels.reshape(-1), ignore_index=-100)
        loss.backward()

        if step == args.iterations - 1 and args.parameter_grad_name:
            parameter = dict(model.named_parameters()).get(args.parameter_grad_name)
            if parameter is None or parameter.grad is None:
                raise RuntimeError(f"parameter has no gradient: {args.parameter_grad_name}")
            parameter_grad = parameter.grad.detach().cpu().float().contiguous().numpy()

        grad_norm = 0.0
        clip_scale = 1.0
        if optimizer is not None:
            parameters = [parameter for _name, parameter in named_trainable]
            clip_threshold = args.max_grad_norm if args.max_grad_norm > 0 else float("inf")
            grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, clip_threshold).detach().float().cpu())
            if args.max_grad_norm > 0:
                clip_scale = min(1.0, args.max_grad_norm / (grad_norm + 1e-6))
            optimizer.step()

        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        loss_value = float(loss.detach().float().cpu())
        peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024 * 1024)
        losses.append(loss_value)
        elapsed_ms_per_step.append(elapsed_ms)
        grad_norms.append(grad_norm)
        clip_scales.append(clip_scale)
        peak_allocated_mb_per_step.append(peak_allocated_mb)
        peak_reserved_mb_per_step.append(peak_reserved_mb)
        phase = "warmup" if step < args.warmup_iterations else "measure"
        print(
            f"step {step + 1}/{args.iterations} | phase {phase} | loss {loss_value:.9g} "
            f"| grad norm {grad_norm:.9g} | clip scale {clip_scale:.9g} | {elapsed_ms:.3f} ms "
            f"| peak allocated {peak_allocated_mb:.1f} MB | peak reserved {peak_reserved_mb:.1f} MB",
            flush=True,
        )
    if hook is not None:
        hook.remove()

    args.output_bin.parent.mkdir(parents=True, exist_ok=True)
    write_sample(args.output_bin, batch["input_ids"], labels, image_indices, flat_vision)
    grad = vision.grad.reshape(-1, vision.shape[-1]).detach().cpu().to(torch.float32).contiguous().numpy()
    args.reference_grad_bin.parent.mkdir(parents=True, exist_ok=True)
    grad.tofile(args.reference_grad_bin)
    if parameter_grad is not None:
        args.parameter_grad_bin.parent.mkdir(parents=True, exist_ok=True)
        parameter_grad.tofile(args.parameter_grad_bin)
    if args.parameter_output_name:
        parameter = dict(model.named_parameters()).get(args.parameter_output_name)
        if parameter is None:
            raise RuntimeError(f"unknown parameter: {args.parameter_output_name}")
        args.parameter_output_bin.parent.mkdir(parents=True, exist_ok=True)
        parameter.detach().cpu().float().contiguous().numpy().tofile(args.parameter_output_bin)
    if args.vpm_output_bin:
        if "hidden_state" not in captured_vpm:
            raise RuntimeError("VPM forward hook did not capture last_hidden_state")
        vpm_hidden = captured_vpm["hidden_state"]
        if vpm_hidden.grad is None:
            raise RuntimeError("VPM output gradient was not retained")
        target_sizes = batch["tgt_sizes"]
        if isinstance(target_sizes, list):
            target_sizes = [item for item in target_sizes if isinstance(item, torch.Tensor)]
            target_sizes = torch.vstack(target_sizes)
        if target_sizes.shape[0] != vpm_hidden.shape[0]:
            raise RuntimeError("only one target size per VPM feature batch is supported")
        if not torch.all(target_sizes == target_sizes[0]):
            raise RuntimeError("VPM alignment export currently requires a uniform target size")
        args.vpm_output_bin.parent.mkdir(parents=True, exist_ok=True)
        write_vpm_sample(args.vpm_output_bin, vpm_hidden, target_sizes[0])
        if args.vpm_reference_grad_bin:
            args.vpm_reference_grad_bin.parent.mkdir(parents=True, exist_ok=True)
            vpm_hidden.grad.detach().cpu().float().contiguous().numpy().tofile(args.vpm_reference_grad_bin)
    measured = slice(args.warmup_iterations, None)
    stats = {
        "dtype": args.dtype,
        "loss": losses[-1],
        "losses": losses,
        "iterations": args.iterations,
        "warmup_iterations": args.warmup_iterations,
        "attention": args.attention,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "adam_beta2": args.adam_beta2,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "tune_vision": args.tune_vision,
        "tune_llm": args.tune_llm,
        "sequence_length": int(batch["input_ids"].shape[1]),
        "visual_tokens": int(flat_vision.shape[0]),
        "hidden_size": int(flat_vision.shape[1]),
        "valid_labels": int((labels != -100).sum().item()),
        "vision_grad_l2": float(torch.linalg.vector_norm(torch.from_numpy(grad)).item()),
        "vision_grad_max_abs": float(np.abs(grad).max()),
        "elapsed_ms": elapsed_ms_per_step[-1],
        "elapsed_ms_per_step": elapsed_ms_per_step,
        "mean_elapsed_ms": float(np.mean(elapsed_ms_per_step[measured])),
        "grad_norms": grad_norms,
        "clip_scales": clip_scales,
        "peak_allocated_mb": float(max(peak_allocated_mb_per_step[measured])),
        "peak_reserved_mb": float(max(peak_reserved_mb_per_step[measured])),
        "peak_allocated_mb_per_step": peak_allocated_mb_per_step,
        "peak_reserved_mb_per_step": peak_reserved_mb_per_step,
        "trainable_parameters": int(sum(parameter.numel() for _name, parameter in named_trainable)),
        "fp32_master_weights": bool(args.fp32_master_weights and dtype != torch.float32),
    }
    if parameter_grad is not None:
        stats["parameter_grad_name"] = args.parameter_grad_name
        stats["parameter_grad_l2"] = float(np.linalg.norm(parameter_grad.reshape(-1)))
        stats["parameter_grad_max_abs"] = float(np.abs(parameter_grad).max())
    if args.vpm_output_bin:
        vpm_grad = captured_vpm["hidden_state"].grad.detach().float()
        stats.update(
            {
                "vpm_tokens": int(captured_vpm["hidden_state"].shape[1]),
                "vpm_hidden_size": int(captured_vpm["hidden_state"].shape[2]),
                "vpm_grad_l2": float(torch.linalg.vector_norm(vpm_grad).cpu()),
                "vpm_grad_max_abs": float(vpm_grad.abs().max().cpu()),
            }
        )
    args.reference_json.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
