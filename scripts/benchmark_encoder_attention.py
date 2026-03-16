import argparse
import csv
import json
import os
import sys
import time
from contextlib import nullcontext

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import vision_transformer as video_vit


def parse_csv_list(raw_value, cast):
    return [cast(v.strip()) for v in raw_value.split(",") if v.strip()]


def resolve_dtype(device, dtype_name):
    if device.type != "cuda":
        return torch.float32
    if dtype_name == "bfloat16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    return torch.float32


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


def make_visible_mask(batch_size, total_tokens, visible_tokens, device):
    return torch.stack(
        [torch.sort(torch.randperm(total_tokens, device=device)[:visible_tokens])[0] for _ in range(batch_size)]
    )


def describe_regime(visible_fraction):
    if visible_fraction <= 0.25:
        return "training_like"
    if visible_fraction >= 0.999:
        return "fully_visible"
    return "lightly_masked"


def autocast_context(device, dtype):
    if device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}:
        return torch.amp.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def benchmark_forward(model, x, mask, dtype, warmup, runs):
    device = x.device

    with torch.no_grad():
        for _ in range(warmup):
            with autocast_context(device, dtype):
                model(x, masks=mask)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        times_ms = []
        for _ in range(runs):
            if device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                with autocast_context(device, dtype):
                    model(x, masks=mask)
                end_event.record()
                torch.cuda.synchronize(device)
                times_ms.append(start_event.elapsed_time(end_event))
            else:
                start_time = time.perf_counter()
                model(x, masks=mask)
                times_ms.append((time.perf_counter() - start_time) * 1000.0)

    times_ms.sort()
    trim = max(1, runs // 10) if runs >= 10 else 0
    trimmed = times_ms[trim:-trim] if trim > 0 else times_ms
    avg_ms = sum(trimmed) / len(trimmed)
    peak_memory_mb = None
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    return avg_ms, peak_memory_mb


def main():
    parser = argparse.ArgumentParser(description="Benchmark V-JEPA 2 encoder attention schedules.")
    parser.add_argument("--model-name", default="vit_large")
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--tubelet-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-frames", default="16,32,64")
    parser.add_argument("--visible-fractions", default="0.25,0.75,1.0")
    parser.add_argument("--mask-ratios", default=None)
    parser.add_argument("--schedules", default="full,12/12,18/6,20/4,area_only")
    parser.add_argument("--area-spatial-splits", type=int, default=2)
    parser.add_argument("--area-temporal-splits", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--json-path", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = resolve_dtype(device, args.dtype)
    num_frames_list = parse_csv_list(args.num_frames, int)
    schedules = parse_csv_list(args.schedules, str)
    if args.mask_ratios is not None:
        visible_fractions = [1.0 - mask_ratio for mask_ratio in parse_csv_list(args.mask_ratios, float)]
    else:
        visible_fractions = parse_csv_list(args.visible_fractions, float)

    constructor = video_vit.__dict__[args.model_name]
    probe_model = constructor(
        img_size=args.crop_size,
        patch_size=args.patch_size,
        num_frames=max(num_frames_list),
        tubelet_size=args.tubelet_size,
        use_rope=True,
        use_sdpa=device.type == "cuda",
    )
    depth = probe_model.get_num_layers()
    del probe_model

    rows = []
    for schedule in schedules:
        attention_pattern = build_attention_pattern(depth, schedule)
        model = constructor(
            img_size=args.crop_size,
            patch_size=args.patch_size,
            num_frames=max(num_frames_list),
            tubelet_size=args.tubelet_size,
            use_rope=True,
            use_sdpa=device.type == "cuda",
            attention_pattern=attention_pattern,
            area_spatial_splits=args.area_spatial_splits,
            area_temporal_splits=args.area_temporal_splits,
        ).to(device=device, dtype=dtype).eval()

        for num_frames in num_frames_list:
            total_tokens = (
                (num_frames // args.tubelet_size)
                * (args.crop_size // args.patch_size)
                * (args.crop_size // args.patch_size)
            )
            x = torch.randn(
                args.batch_size,
                3,
                num_frames,
                args.crop_size,
                args.crop_size,
                device=device,
                dtype=dtype,
            )

            for visible_fraction in visible_fractions:
                visible_tokens = max(1, int(total_tokens * visible_fraction))
                visible_fraction = visible_tokens / total_tokens
                mask_ratio = 1.0 - visible_fraction
                mask = None if visible_tokens == total_tokens else make_visible_mask(
                    args.batch_size, total_tokens, visible_tokens, device
                )
                avg_ms, peak_memory_mb = benchmark_forward(
                    model=model,
                    x=x,
                    mask=mask,
                    dtype=dtype,
                    warmup=args.warmup,
                    runs=args.runs,
                )
                rows.append(
                    {
                        "scope": "encoder_only",
                        "model_name": args.model_name,
                        "schedule": schedule,
                        "regime": describe_regime(visible_fraction),
                        "num_frames": num_frames,
                        "total_tokens": total_tokens,
                        "visible_fraction": round(visible_fraction, 4),
                        "mask_ratio": mask_ratio,
                        "visible_tokens": visible_tokens,
                        "latency_ms": round(avg_ms, 4),
                        "throughput_videos_per_s": round(args.batch_size / (avg_ms / 1000.0), 4),
                        "peak_memory_mb": None if peak_memory_mb is None else round(peak_memory_mb, 2),
                    }
                )
                print(rows[-1])

            del x
            if device.type == "cuda":
                torch.cuda.empty_cache()

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

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
