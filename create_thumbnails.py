# generation/generate_from_ckpt.py
import os
import math
import json
import csv
import time
import argparse
import re
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import psutil
import torch
from torchvision.utils import save_image, make_grid

from train_UNet.model import DiffusionUNetLit, DiffusionUNetLitNoAttn
from train_UNet.dataset import DiffusionSchedule


# =========================================================
# SAMPLE CALL
# =========================================================
# python -m time_and_memory_test.py \
#   --exp exp7 \
#   --ckpt_dir /storage/.../checkpoints \
#   --outdir /storage/.../gen_out \
#   --num_samples 64 \
#   --batch_size 8 \
#   --sampler ddim \
#   --sample_steps 50 \
#   --eta 0.0 \
#   --attention false \
#   --seed 42 \
#   --profile_every_batch
# =========================================================


def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "t", "1", "yes", "y", "on"):
        return True
    if s in ("false", "f", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: '{v}' (use true/false)")


def infer_ckpt_step(ckpt_path: str) -> int | None:
    s = str(ckpt_path)
    patterns = [
        r"stepstep\\?=([0-9]+)",
        r"(?:^|[^a-zA-Z])step\\?=([0-9]+)",
        r"global[_\-]?step\\?=([0-9]+)",
        r"global[_\-]?step([0-9]+)",
        r"(?:^|[^a-zA-Z])step[_\-]?([0-9]+)",
    ]
    for pat in patterns:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def _sanitize_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)


def list_checkpoints(ckpt_dir: str) -> list[Path]:
    p = Path(ckpt_dir)
    if not p.is_dir():
        raise SystemExit(f"[ERROR] --ckpt_dir is not a directory: {ckpt_dir}")

    ckpts = sorted(p.glob("*.ckpt"))
    if not ckpts:
        raise SystemExit(f"[ERROR] No .ckpt files found in: {ckpt_dir}")

    best = []
    stepped = []
    for c in ckpts:
        st = infer_ckpt_step(str(c))
        if st is None:
            best.append(c)
        else:
            stepped.append((st, c))

    stepped = [c for _, c in sorted(stepped, key=lambda x: x[0])]
    best = sorted(best)
    return best + stepped


def make_timestep_schedule(T: int, sample_steps: int) -> list[int]:
    if sample_steps >= T:
        return list(range(T - 1, -1, -1))

    ts = np.linspace(0, T - 1, sample_steps, dtype=np.int64)
    ts = np.unique(ts)
    ts = ts[::-1].tolist()
    if ts[-1] != 0:
        ts.append(0)
    return ts


def get_ram_usage_mb() -> float:
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / (1024 ** 2)


def get_gpu_mem_stats(device: torch.device) -> dict:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "gpu_allocated_mb": 0.0,
            "gpu_reserved_mb": 0.0,
            "gpu_max_allocated_mb": 0.0,
            "gpu_max_reserved_mb": 0.0,
            "gpu_free_mb": 0.0,
            "gpu_total_mb": 0.0,
        }

    idx = device.index if device.index is not None else torch.cuda.current_device()
    free_b, total_b = torch.cuda.mem_get_info(idx)
    return {
        "gpu_allocated_mb": torch.cuda.memory_allocated(device) / (1024 ** 2),
        "gpu_reserved_mb": torch.cuda.memory_reserved(device) / (1024 ** 2),
        "gpu_max_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
        "gpu_max_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024 ** 2),
        "gpu_free_mb": free_b / (1024 ** 2),
        "gpu_total_mb": total_b / (1024 ** 2),
    }


@torch.no_grad()
def predict_eps(
    lit_model: DiffusionUNetLit | DiffusionUNetLitNoAttn,
    x_t: torch.Tensor,
    t: torch.Tensor,
    class_id: torch.Tensor | None,
    guidance_scale: float,
) -> torch.Tensor:
    denoiser = lit_model.model

    if (class_id is None) or (guidance_scale == 1.0):
        return denoiser(x_t, time=t, classes=class_id, cond_drop_prob=0.0)

    eps_uncond = denoiser(x_t, time=t, classes=None, cond_drop_prob=0.0)
    eps_cond = denoiser(x_t, time=t, classes=class_id, cond_drop_prob=0.0)
    return eps_uncond + guidance_scale * (eps_cond - eps_uncond)


@torch.no_grad()
def ddpm_step(
    lit_model: DiffusionUNetLit | DiffusionUNetLitNoAttn,
    schedule: DiffusionSchedule,
    x: torch.Tensor,
    t_idx: int,
    class_id: int | None,
    guidance_scale: float,
    rand_gen: torch.Generator | None,
) -> torch.Tensor:
    device = x.device
    bs = x.shape[0]

    betas = schedule.betas.to(device)
    alphas = schedule.alphas.to(device)
    abar = schedule.alphas_cumprod.to(device)

    t = torch.full((bs,), t_idx, device=device, dtype=torch.long)
    y = None if class_id is None else torch.full((bs,), int(class_id), device=device, dtype=torch.long)

    beta_t = betas[t_idx]
    alpha_t = alphas[t_idx]
    abar_t = abar[t_idx]
    abar_prev = abar[t_idx - 1] if t_idx > 0 else torch.tensor(1.0, device=device)

    eps = predict_eps(lit_model, x, t, y, guidance_scale=guidance_scale)
    mu = (1.0 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1.0 - abar_t)) * eps)

    if t_idx == 0:
        return mu

    var = beta_t * (1.0 - abar_prev) / (1.0 - abar_t)
    noise = torch.randn(
        x.shape,
        device=device,
        dtype=x.dtype,
        generator=rand_gen,
    )
    return mu + torch.sqrt(var) * noise


@torch.no_grad()
def ddim_step(
    lit_model: DiffusionUNetLit | DiffusionUNetLitNoAttn,
    schedule: DiffusionSchedule,
    x: torch.Tensor,
    t_idx: int,
    t_prev: int,
    class_id: int | None,
    guidance_scale: float,
    eta: float,
    clip_denoised: bool,
    rand_gen: torch.Generator | None,
) -> torch.Tensor:
    device = x.device
    bs = x.shape[0]

    abar = schedule.alphas_cumprod.to(device)
    t = torch.full((bs,), t_idx, device=device, dtype=torch.long)
    y = None if class_id is None else torch.full((bs,), int(class_id), device=device, dtype=torch.long)

    abar_t = abar[t_idx]
    abar_prev = abar[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)

    eps = predict_eps(lit_model, x, t, y, guidance_scale=guidance_scale)

    sqrt_abar_t = torch.sqrt(abar_t)
    sqrt_one_minus_abar_t = torch.sqrt(1.0 - abar_t)
    x0 = (x - sqrt_one_minus_abar_t * eps) / sqrt_abar_t

    if clip_denoised:
        x0 = x0.clamp(-1.0, 1.0)

    sigma = eta * torch.sqrt((1.0 - abar_prev) / (1.0 - abar_t)) * torch.sqrt(1.0 - (abar_t / abar_prev))
    dir_coeff = torch.sqrt(torch.clamp(1.0 - abar_prev - sigma ** 2, min=0.0))
    x_prev = torch.sqrt(abar_prev) * x0 + dir_coeff * eps

    if t_prev >= 0 and eta > 0.0:
        noise = torch.randn(
            x.shape,
            device=device,
            dtype=x.dtype,
            generator=rand_gen,
        )
        x_prev = x_prev + sigma * noise

    return x_prev


def estimate_latent_bank_ram_mb(num_samples: int, channels: int, image_size: int, bytes_per_elem: int = 4) -> float:
    n_elems = num_samples * channels * image_size * image_size
    return (n_elems * bytes_per_elem) / (1024 ** 2)


def autocast_context(device: torch.device, enabled: bool, dtype_str: str):
    if not enabled or device.type != "cuda":
        return nullcontext()

    if dtype_str == "float16":
        dtype = torch.float16
    elif dtype_str == "bfloat16":
        dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported autocast dtype: {dtype_str}")

    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.no_grad()
def sequential_streaming_sample(
    lit_model: DiffusionUNetLit | DiffusionUNetLitNoAttn,
    schedule: DiffusionSchedule,
    args,
    device: torch.device,
    out_dir: str,
    ckpt_stem: str,
    step_tag: str,
):
    total_samples = args.num_samples if args.num_samples is not None else args.n
    batch_size = args.batch_size if args.batch_size is not None else min(args.n, total_samples)

    if args.sampler == "ddpm":
        timesteps = list(range(schedule.num_timesteps - 1, -1, -1))
    else:
        timesteps = make_timestep_schedule(schedule.num_timesteps, args.sample_steps)

    os.makedirs(out_dir, exist_ok=True)
    profile_dir = os.path.join(out_dir, "profiling")
    os.makedirs(profile_dir, exist_ok=True)

    estimated_bank_mb = estimate_latent_bank_ram_mb(
        num_samples=total_samples,
        channels=args.channels,
        image_size=args.image_size,
        bytes_per_elem=4,
    )

    print(f"[INFO] Streaming sequential sampling")
    print(f"[INFO] Total samples: {total_samples}")
    print(f"[INFO] Batch size: {batch_size}")
    print(f"[INFO] Passes: {len(timesteps)}")
    print(f"[INFO] Estimated latent CPU bank size: {estimated_bank_mb:.2f} MB")

    # deterministic generators
    cpu_gen = torch.Generator(device="cpu")
    cpu_gen.manual_seed(args.seed)

    dev_gen = None
    if device.type == "cuda":
        dev_gen = torch.Generator(device=f"cuda:{device.index if device.index is not None else torch.cuda.current_device()}")
        dev_gen.manual_seed(args.seed + 1)

    # CPU latent bank
    pin = bool(device.type == "cuda")
    x_bank_cpu = torch.empty(
        (total_samples, args.channels, args.image_size, args.image_size),
        dtype=torch.float32,
        pin_memory=pin,
    )
    x_bank_cpu.normal_(generator=cpu_gen)

    step_csv = os.path.join(profile_dir, f"{args.exp}_{ckpt_stem}_{step_tag}_step_profile.csv")
    batch_csv = os.path.join(profile_dir, f"{args.exp}_{ckpt_stem}_{step_tag}_batch_profile.csv")
    summary_json = os.path.join(profile_dir, f"{args.exp}_{ckpt_stem}_{step_tag}_summary.json")

    step_rows = []
    batch_rows = []

    t_global0 = time.perf_counter()

    for pass_idx, t_idx in enumerate(timesteps):
        t_prev = timesteps[pass_idx + 1] if (args.sampler == "ddim" and (pass_idx + 1) < len(timesteps)) else -1

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        step_wall0 = time.perf_counter()
        step_h2d = 0.0
        step_compute = 0.0
        step_d2h = 0.0

        step_ram_before = get_ram_usage_mb()
        step_gpu_before = get_gpu_mem_stats(device)

        print(f"[STEP {pass_idx + 1:04d}/{len(timesteps):04d}] t={t_idx} started...")

        for batch_idx, start in enumerate(range(0, total_samples, batch_size)):
            end = min(start + batch_size, total_samples)
            bs = end - start

            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)

            batch_wall0 = time.perf_counter()
            ram_before = get_ram_usage_mb()
            gpu_before = get_gpu_mem_stats(device)

            # H2D
            t0 = time.perf_counter()
            x = x_bank_cpu[start:end].to(device, non_blocking=pin)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            h2d_s = time.perf_counter() - t0

            # compute
            t1 = time.perf_counter()
            with autocast_context(device, args.use_autocast, args.autocast_dtype):
                if args.sampler == "ddpm":
                    x_next = ddpm_step(
                        lit_model=lit_model,
                        schedule=schedule,
                        x=x,
                        t_idx=t_idx,
                        class_id=args.class_id,
                        guidance_scale=args.guidance_scale,
                        rand_gen=dev_gen,
                    )
                else:
                    x_next = ddim_step(
                        lit_model=lit_model,
                        schedule=schedule,
                        x=x,
                        t_idx=t_idx,
                        t_prev=t_prev,
                        class_id=args.class_id,
                        guidance_scale=args.guidance_scale,
                        eta=args.eta,
                        clip_denoised=args.clip_denoised,
                        rand_gen=dev_gen,
                    )

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            compute_s = time.perf_counter() - t1

            # D2H
            t2 = time.perf_counter()
            x_bank_cpu[start:end].copy_(x_next.detach().to("cpu", non_blocking=False))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            d2h_s = time.perf_counter() - t2

            batch_wall_s = time.perf_counter() - batch_wall0
            ram_after = get_ram_usage_mb()
            gpu_after = get_gpu_mem_stats(device)

            step_h2d += h2d_s
            step_compute += compute_s
            step_d2h += d2h_s

            if args.profile_every_batch:
                batch_rows.append({
                    "pass_idx": pass_idx,
                    "t_idx": t_idx,
                    "t_prev": t_prev,
                    "batch_idx": batch_idx,
                    "start": start,
                    "end": end,
                    "batch_size": bs,
                    "wall_s": batch_wall_s,
                    "h2d_s": h2d_s,
                    "compute_s": compute_s,
                    "d2h_s": d2h_s,
                    "ram_before_mb": ram_before,
                    "ram_after_mb": ram_after,
                    "gpu_alloc_before_mb": gpu_before["gpu_allocated_mb"],
                    "gpu_alloc_after_mb": gpu_after["gpu_allocated_mb"],
                    "gpu_reserved_before_mb": gpu_before["gpu_reserved_mb"],
                    "gpu_reserved_after_mb": gpu_after["gpu_reserved_mb"],
                    "gpu_peak_alloc_mb": gpu_after["gpu_max_allocated_mb"],
                    "gpu_peak_reserved_mb": gpu_after["gpu_max_reserved_mb"],
                    "gpu_free_after_mb": gpu_after["gpu_free_mb"],
                })

            del x, x_next

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        step_wall_s = time.perf_counter() - step_wall0
        step_ram_after = get_ram_usage_mb()
        step_gpu_after = get_gpu_mem_stats(device)

        step_rows.append({
            "pass_idx": pass_idx,
            "t_idx": t_idx,
            "t_prev": t_prev,
            "num_batches": math.ceil(total_samples / batch_size),
            "wall_s": step_wall_s,
            "h2d_total_s": step_h2d,
            "compute_total_s": step_compute,
            "d2h_total_s": step_d2h,
            "ram_before_mb": step_ram_before,
            "ram_after_mb": step_ram_after,
            "gpu_alloc_before_mb": step_gpu_before["gpu_allocated_mb"],
            "gpu_alloc_after_mb": step_gpu_after["gpu_allocated_mb"],
            "gpu_reserved_before_mb": step_gpu_before["gpu_reserved_mb"],
            "gpu_reserved_after_mb": step_gpu_after["gpu_reserved_mb"],
            "gpu_peak_alloc_mb": step_gpu_after["gpu_max_allocated_mb"],
            "gpu_peak_reserved_mb": step_gpu_after["gpu_max_reserved_mb"],
            "gpu_free_after_mb": step_gpu_after["gpu_free_mb"],
        })

        print(
            f"[STEP {pass_idx + 1:04d}/{len(timesteps):04d}] "
            f"t={t_idx} done | wall={step_wall_s:.3f}s | "
            f"h2d={step_h2d:.3f}s | compute={step_compute:.3f}s | d2h={step_d2h:.3f}s | "
            f"RAM={step_ram_after:.1f}MB | GPU peak={step_gpu_after['gpu_max_allocated_mb']:.1f}MB"
        )

    total_wall_s = time.perf_counter() - t_global0

    # Save samples only at the end
    save0 = time.perf_counter()
    preview_cache = []

    for i in range(total_samples):
        img = (x_bank_cpu[i].clamp(-1, 1) + 1.0) * 0.5
        fname = f"{args.exp}_{ckpt_stem}_{step_tag}_sample_{i:06d}.png"
        out_path = os.path.join(out_dir, fname)
        save_image(img, out_path)

        if args.save_grid and (args.preview_max is None or len(preview_cache) < args.preview_max):
            preview_cache.append(img.unsqueeze(0))

    save_wall_s = time.perf_counter() - save0

    if args.save_grid and len(preview_cache) > 0:
        all_imgs = torch.cat(preview_cache, dim=0)
        n = all_imgs.shape[0]
        nrow = max(1, min(int(math.sqrt(n)), 16))
        grid = make_grid(all_imgs, nrow=nrow)
        grid_path = os.path.join(
            args.outdir,
            args.exp,
            f"{args.exp}_{ckpt_stem}_{step_tag}_preview_{args.sampler}_passes{len(timesteps)}_n{n}.png",
        )
        save_image(grid, grid_path)
        print(f"[OK] Saved preview grid: {grid_path}")

    # Write CSVs
    with open(step_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(step_rows[0].keys()))
        writer.writeheader()
        writer.writerows(step_rows)

    if args.profile_every_batch and len(batch_rows) > 0:
        with open(batch_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(batch_rows[0].keys()))
            writer.writeheader()
            writer.writerows(batch_rows)

    summary = {
        "exp": args.exp,
        "checkpoint": str(ckpt_stem),
        "step_tag": step_tag,
        "sampler": args.sampler,
        "num_samples": total_samples,
        "batch_size": batch_size,
        "image_size": args.image_size,
        "channels": args.channels,
        "num_passes": len(timesteps),
        "timesteps": timesteps,
        "estimated_latent_bank_ram_mb": estimated_bank_mb,
        "total_sampling_wall_s": total_wall_s,
        "final_save_wall_s": save_wall_s,
        "final_ram_mb": get_ram_usage_mb(),
        "final_gpu_stats": get_gpu_mem_stats(device),
        "profile_every_batch": args.profile_every_batch,
        "use_autocast": args.use_autocast,
        "autocast_dtype": args.autocast_dtype if args.use_autocast else None,
    }

    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] Saved step profile: {step_csv}")
    if args.profile_every_batch and len(batch_rows) > 0:
        print(f"[OK] Saved batch profile: {batch_csv}")
    print(f"[OK] Saved summary: {summary_json}")
    print(f"[OK] Saved {total_samples}/{total_samples} samples to {out_dir}")

    del x_bank_cpu
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_one_checkpoint(args, ckpt_path: str, schedule: DiffusionSchedule, device: torch.device):
    ckpt_path = str(ckpt_path)
    ckpt_file = Path(ckpt_path)

    ckpt_step = args.ckpt_step if args.ckpt_step is not None else infer_ckpt_step(ckpt_path)
    step_tag = f"step{ckpt_step}" if ckpt_step is not None else "best"

    ckpt_stem = _sanitize_name(ckpt_file.stem)
    out_dir = os.path.join(args.outdir, args.exp, ckpt_stem)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n[CKPT] {ckpt_path}")
    print(f"[OUT ] {out_dir}")
    print(f"[ARCH] attention={args.attention}")

    model_cls = DiffusionUNetLit if args.attention else DiffusionUNetLitNoAttn
    lit_model = model_cls.load_from_checkpoint(ckpt_path, map_location=device)
    lit_model.eval()
    lit_model.to(device)

    sequential_streaming_sample(
        lit_model=lit_model,
        schedule=schedule,
        args=args,
        device=device,
        out_dir=out_dir,
        ckpt_stem=ckpt_stem,
        step_tag=step_tag,
    )

    del lit_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", type=str, required=True, help="Exp number (e.g., exp7)")

    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--ckpt", type=str, help="Path to a single .ckpt")
    grp.add_argument("--ckpt_dir", type=str, help="Folder containing multiple .ckpt files")

    ap.add_argument("--ckpt_step", type=int, default=None, help="Override ckpt step (single --ckpt mode)")

    ap.add_argument("--outdir", type=str, default="./gen_out")
    ap.add_argument("--n", type=int, default=16, help="Deprecated fallback")
    ap.add_argument("--num_samples", type=int, default=None, help="Total number of samples to generate per checkpoint")
    ap.add_argument("--batch_size", type=int, default=None, help="Streaming chunk size per denoising update")

    ap.add_argument("--save_grid", action="store_true", help="Save preview grid from final samples")
    ap.add_argument("--preview_max", type=int, default=64, help="Max images in preview grid")

    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--channels", type=int, default=3)

    ap.add_argument("--num_timesteps", type=int, default=1000)
    ap.add_argument("--schedule", type=str, default="linear", choices=["linear", "cosine"])
    ap.add_argument("--beta_start", type=float, default=1e-4)
    ap.add_argument("--beta_end", type=float, default=2e-2)
    ap.add_argument("--cosine_s", type=float, default=0.008)

    ap.add_argument("--sampler", type=str, default="ddim", choices=["ddim", "ddpm"])
    ap.add_argument("--sample_steps", type=int, default=50, help="Used by DDIM only")
    ap.add_argument("--eta", type=float, default=0.0, help="DDIM stochasticity")
    ap.add_argument("--clip_denoised", action="store_true", help="Clamp predicted x0 to [-1,1] during DDIM")

    ap.add_argument("--class_id", type=int, default=None, help="Conditional class id")
    ap.add_argument("--guidance_scale", type=float, default=1.0, help="CFG scale")

    ap.add_argument(
        "--attention",
        type=str2bool,
        default=True,
        help="true: DiffusionUNetLit, false: DiffusionUNetLitNoAttn",
    )

    ap.add_argument("--seed", type=int, default=42, help="Random seed for initial noise and stochastic sampling")
    ap.add_argument("--profile_every_batch", action="store_true", help="Also save per-microbatch timings and memory")
    ap.add_argument("--use_autocast", action="store_true", help="Use CUDA autocast during inference")
    ap.add_argument("--autocast_dtype", type=str, default="float16", choices=["float16", "bfloat16"])

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, args.exp), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    schedule = DiffusionSchedule(
        num_timesteps=args.num_timesteps,
        schedule=args.schedule,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        cosine_s=args.cosine_s,
    )

    if args.ckpt_dir:
        ckpts = list_checkpoints(args.ckpt_dir)
        print(f"[INFO] Found {len(ckpts)} checkpoints in {args.ckpt_dir}")
        for c in ckpts:
            run_one_checkpoint(args, str(c), schedule, device)
    else:
        run_one_checkpoint(args, args.ckpt, schedule, device)


if __name__ == "__main__":
    main()