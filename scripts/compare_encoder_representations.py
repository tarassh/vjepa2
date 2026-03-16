import argparse
import csv
import json
import os
import sys
from contextlib import nullcontext

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import vision_transformer as video_vit


def clean_state_dict(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        key = key.replace("module.", "")
        key = key.replace("backbone.", "")
        cleaned[key] = value
    return cleaned


def extract_encoder_state_dict(checkpoint_path, checkpoint_key):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint_key is None:
        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint must be a dict when checkpoint_key is omitted")
        state_dict = checkpoint
    else:
        state_dict = checkpoint[checkpoint_key]
    return clean_state_dict(state_dict)


def build_attention_pattern(depth, schedule):
    schedule = schedule.strip().lower()
    if schedule == "full":
        return ["global"] * depth
    if schedule == "area_only":
        return ["area"] * depth
    if "/" in schedule:
        local_layers, global_layers = (int(v) for v in schedule.split("/", maxsplit=1))
        if local_layers + global_layers != depth:
            raise ValueError(f"Schedule {schedule!r} does not match depth={depth}")
        return ["area"] * local_layers + ["global"] * global_layers
    raise ValueError(f"Unsupported schedule {schedule!r}")


def resolve_dtype(device, dtype_name):
    if device.type != "cuda":
        return torch.float32
    if dtype_name == "bfloat16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    return torch.float32


def autocast_context(device, dtype):
    if device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}:
        return torch.amp.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def make_visible_mask(batch_size, total_tokens, visible_tokens, device):
    return torch.stack(
        [torch.sort(torch.randperm(total_tokens, device=device)[:visible_tokens])[0] for _ in range(batch_size)]
    )


def linear_cka(x, y):
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    xxt = x.T @ x
    yyt = y.T @ y
    xyt = x.T @ y
    numerator = (xyt * xyt).sum()
    denominator = torch.sqrt((xxt * xxt).sum() * (yyt * yyt).sum()).clamp_min(1.0e-12)
    return numerator / denominator


def build_model(args, num_frames, attention_pattern, device, dtype):
    constructor = video_vit.__dict__[args.model_name]
    return constructor(
        img_size=args.crop_size,
        patch_size=args.patch_size,
        num_frames=num_frames,
        tubelet_size=args.tubelet_size,
        use_rope=True,
        use_sdpa=device.type == "cuda",
        attention_pattern=attention_pattern,
        area_spatial_splits=args.area_spatial_splits,
        area_temporal_splits=args.area_temporal_splits,
    ).to(device=device, dtype=dtype).eval()


def load_clips(args, device, dtype):
    if args.clips_path is not None:
        clips = torch.load(args.clips_path, map_location="cpu")
        clip_source = args.clips_path
    else:
        torch.manual_seed(args.seed)
        clips = torch.randn(args.batch_size, 3, args.num_frames, args.crop_size, args.crop_size)
        clip_source = "synthetic_random"
    return clips.to(device=device, dtype=dtype), clip_source


def main():
    parser = argparse.ArgumentParser(description="Encoder-only representation retention comparison.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-key", default="target_encoder")
    parser.add_argument("--model-name", default="vit_large")
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--tubelet-size", type=int, default=2)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--visible-fraction", type=float, default=1.0)
    parser.add_argument("--schedules", default="12/12,18/6,20/4")
    parser.add_argument("--area-spatial-splits", type=int, default=2)
    parser.add_argument("--area-temporal-splits", type=int, default=2)
    parser.add_argument("--clips-path", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--json-path", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = resolve_dtype(device, args.dtype)
    schedules = [s.strip() for s in args.schedules.split(",") if s.strip()]

    constructor = video_vit.__dict__[args.model_name]
    probe = constructor(
        img_size=args.crop_size,
        patch_size=args.patch_size,
        num_frames=args.num_frames,
        tubelet_size=args.tubelet_size,
        use_rope=True,
        use_sdpa=device.type == "cuda",
    )
    depth = probe.get_num_layers()
    del probe
    baseline = build_model(args, args.num_frames, ["global"] * depth, device, dtype)

    checkpoint_state_dict = extract_encoder_state_dict(args.checkpoint, args.checkpoint_key)
    baseline_load_msg = baseline.load_state_dict(checkpoint_state_dict, strict=False)
    baseline_runtime_state_dict = baseline.state_dict()

    clips, clip_source = load_clips(args, device, dtype)
    total_tokens = (args.num_frames // args.tubelet_size) * (args.crop_size // args.patch_size) ** 2
    visible_tokens = max(1, int(total_tokens * args.visible_fraction))
    mask = None if visible_tokens == total_tokens else make_visible_mask(args.batch_size, total_tokens, visible_tokens, device)

    with torch.no_grad():
        with autocast_context(device, dtype):
            baseline_out = baseline(clips, masks=mask) if mask is not None else baseline(clips)

    rows = []
    for schedule in schedules:
        hybrid = build_model(args, args.num_frames, build_attention_pattern(depth, schedule), device, dtype)
        hybrid_load_msg = hybrid.load_state_dict(baseline_runtime_state_dict, strict=True)
        with torch.no_grad():
            with autocast_context(device, dtype):
                hybrid_out = hybrid(clips, masks=mask) if mask is not None else hybrid(clips)

        flat_baseline = baseline_out.float().reshape(-1, baseline_out.shape[-1])
        flat_hybrid = hybrid_out.float().reshape(-1, hybrid_out.shape[-1])
        pooled_baseline = baseline_out.float().mean(dim=1)
        pooled_hybrid = hybrid_out.float().mean(dim=1)

        row = {
            "scope": "encoder_only",
            "schedule": schedule,
            "num_frames": args.num_frames,
            "visible_tokens": visible_tokens,
            "checkpoint": args.checkpoint,
            "checkpoint_key": args.checkpoint_key,
            "clip_source": clip_source,
            "baseline_checkpoint_missing_keys": baseline_load_msg.missing_keys,
            "baseline_checkpoint_unexpected_keys": baseline_load_msg.unexpected_keys,
            "hybrid_strict_missing_keys": hybrid_load_msg.missing_keys,
            "hybrid_strict_unexpected_keys": hybrid_load_msg.unexpected_keys,
            "token_cosine_mean": round(
                F.cosine_similarity(flat_baseline, flat_hybrid, dim=-1).mean().item(),
                6,
            ),
            "pooled_cosine_mean": round(
                F.cosine_similarity(pooled_baseline, pooled_hybrid, dim=-1).mean().item(),
                6,
            ),
            "linear_cka": round(linear_cka(flat_baseline, flat_hybrid).item(), 6),
        }
        rows.append(row)
        print(row)

    if args.csv_path is not None:
        with open(args.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    if args.json_path is not None:
        with open(args.json_path, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
