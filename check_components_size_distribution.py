# generation/generate_from_ckpt.py
import os
import math
import argparse
import re
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image, make_grid

from train_UNet.model import DiffusionUNetLit, DiffusionUNetLitNoAttn
from train_UNet.dataset import DiffusionSchedule

## SAMPLE CALL
# python -m generation.generate_from_ckpt \
#   --exp exp7 \
#   --ckpt_dir /storage/.../checkpoints \
#   --outdir /storage/.../gen_out \
#   --num_samples 64 --batch_size 16 \
#   --sampler ddim --sample_steps 50 --eta 0.0 \
#   --attention false
##

def str2bool(v):
    """Parse true/false from CLI."""
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "t", "1", "yes", "y", "on"):
        return True
    if s in ("false", "f", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: '{v}' (use true/false)")


def infer_ckpt_step(ckpt_path: str) -> int | None:
    """
    Infer step number from checkpoint path/name.

    Handles patterns like:
      ...-stepstep\\=60000.ckpt   (your case)
      ...-stepstep=60000.ckpt
      ...-step=60000.ckpt
      ...global_step60000.ckpt
      ...step60000.ckpt / step_60000.ckpt
    """
    s = str(ckpt_path)

    patterns = [
        r"stepstep\\?=([0-9]+)",               # stepstep\=60000 or stepstep=60000
        r"(?:^|[^a-zA-Z])step\\?=([0-9]+)",    # step\=60000 or step=60000 (guarded)
        r"global[_\-]?step\\?=([0-9]+)",       # global_step=60000 (rare)
        r"global[_\-]?step([0-9]+)",           # global_step60000
        r"(?:^|[^a-zA-Z])step[_\-]?([0-9]+)",  # step60000 / step_60000 / -step60000
    ]

    for pat in patterns:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))

    return None


def _sanitize_name(s: str) -> str:
    # safe folder/file component
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)


def list_checkpoints(ckpt_dir: str) -> list[Path]:
    """
    List checkpoints in a folder. Assumes the *best* checkpoint is the one
    that does NOT have a step pattern in the name (i.e., infer_ckpt_step == None).
    """
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


@torch.no_grad()
def predict_eps(
    lit_model: DiffusionUNetLit | DiffusionUNetLitNoAttn,
    x_t: torch.Tensor,
    t: torch.Tensor,
    class_id: torch.Tensor | None,
    guidance_scale: float
) -> torch.Tensor:
    """
    Classifier-free guidance (CFG).
    We call lit_model.model(...) directly with cond_drop_prob=0 to disable dropout at sampling time.
    """
    denoiser = lit_model.model  # UNetDenoiser

    if (class_id is None) or (guidance_scale == 1.0):
        return denoiser(x_t, time=t, classes=class_id, cond_drop_prob=0.0)

    eps_uncond = denoiser(x_t, time=t, classes=None, cond_drop_prob=0.0)
    eps_cond   = denoiser(x_t, time=t, classes=class_id, cond_drop_prob=0.0)
    return eps_uncond + guidance_scale * (eps_cond - eps_uncond)


def make_timestep_schedule(T: int, sample_steps: int) -> list[int]:
    """Returns descending list of timesteps to traverse."""
    if sample_steps >= T:
        return list(range(T - 1, -1, -1))

    ts = np.linspace(0, T - 1, sample_steps, dtype=np.int64)
    ts = np.unique(ts)
    ts = ts[::-1].tolist()
    if ts[-1] != 0:
        ts.append(0)
    return ts


@torch.no_grad()
def ddpm_sample_full(
    lit_model: DiffusionUNetLit | DiffusionUNetLitNoAttn,
    schedule: DiffusionSchedule,
    batch_size: int,
    image_size: int,
    channels: int,
    class_id: int | None,
    guidance_scale: float,
    device: torch.device
) -> torch.Tensor:
    """Classic DDPM ancestral sampling (FULL steps)."""
    T = schedule.num_timesteps
    betas  = schedule.betas.to(device)
    alphas = schedule.alphas.to(device)
    abar   = schedule.alphas_cumprod.to(device)

    x = torch.randn(batch_size, channels, image_size, image_size, device=device)
    y = None if class_id is None else torch.full((batch_size,), int(class_id), device=device, dtype=torch.long)

    for t_idx in range(T - 1, -1, -1):
        t = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)

        beta_t = betas[t_idx]
        alpha_t = alphas[t_idx]
        abar_t = abar[t_idx]
        abar_prev = abar[t_idx - 1] if t_idx > 0 else torch.tensor(1.0, device=device)

        eps = predict_eps(lit_model, x, t, y, guidance_scale=guidance_scale)

        mu = (1.0 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1.0 - abar_t)) * eps)

        if t_idx == 0:
            x = mu
            continue

        var = beta_t * (1.0 - abar_prev) / (1.0 - abar_t)
        x = mu + torch.sqrt(var) * torch.randn_like(x)

    return x


@torch.no_grad()
def ddim_sample(
    lit_model: DiffusionUNetLit | DiffusionUNetLitNoAttn,
    schedule: DiffusionSchedule,
    batch_size: int,
    image_size: int,
    channels: int,
    class_id: int | None,
    guidance_scale: float,
    sample_steps: int,
    eta: float,
    clip_denoised: bool,
    device: torch.device
) -> torch.Tensor:
    """
    DDIM sampling with timestep respacing.
    - sample_steps controls #passes (model calls)
    - eta=0.0 -> deterministic DDIM
    - eta>0.0 -> adds stochasticity
    """
    T = schedule.num_timesteps
    abar = schedule.alphas_cumprod.to(device)

    timesteps = make_timestep_schedule(T, sample_steps)

    x = torch.randn(batch_size, channels, image_size, image_size, device=device)
    y = None if class_id is None else torch.full((batch_size,), int(class_id), device=device, dtype=torch.long)

    for i, t_idx in enumerate(timesteps):
        t_prev = timesteps[i + 1] if (i + 1) < len(timesteps) else -1
        t = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)

        abar_t = abar[t_idx]
        abar_prev = abar[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)

        eps = predict_eps(lit_model, x, t, y, guidance_scale=guidance_scale)

        sqrt_abar_t = torch.sqrt(abar_t)
        sqrt_one_minus_abar_t = torch.sqrt(1.0 - abar_t)
        x0 = (x - sqrt_one_minus_abar_t * eps) / sqrt_abar_t

        if clip_denoised:
            x0 = x0.clamp(-1.0, 1.0)

        sigma = eta * torch.sqrt((1.0 - abar_prev) / (1.0 - abar_t)) * torch.sqrt(1.0 - (abar_t / abar_prev))
        dir_coeff = torch.sqrt(torch.clamp(1.0 - abar_prev - sigma**2, min=0.0))
        x_prev = torch.sqrt(abar_prev) * x0 + dir_coeff * eps

        if t_prev >= 0 and eta > 0.0:
            x_prev = x_prev + sigma * torch.randn_like(x)

        x = x_prev

    return x


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

    total_samples = args.num_samples if args.num_samples is not None else args.n
    batch_size = args.batch_size if args.batch_size is not None else min(args.n, total_samples)

    def _sample_batch(bs: int) -> torch.Tensor:
        if args.sampler == "ddpm":
            return ddpm_sample_full(
                lit_model=lit_model,
                schedule=schedule,
                batch_size=bs,
                image_size=args.image_size,
                channels=args.channels,
                class_id=args.class_id,
                guidance_scale=args.guidance_scale,
                device=device,
            )
        return ddim_sample(
            lit_model=lit_model,
            schedule=schedule,
            batch_size=bs,
            image_size=args.image_size,
            channels=args.channels,
            class_id=args.class_id,
            guidance_scale=args.guidance_scale,
            sample_steps=args.sample_steps,
            eta=args.eta,
            clip_denoised=args.clip_denoised,
            device=device,
        )

    passes = args.num_timesteps if args.sampler == "ddpm" else args.sample_steps
    saved = 0

    preview_batches_done = 0
    preview_cache: list[torch.Tensor] = []
    preview_saved = False

    while saved < total_samples:
        bs = min(batch_size, total_samples - saved)
        samples = _sample_batch(bs)
        samples_01 = (samples.clamp(-1, 1) + 1) * 0.5

        for i in range(bs):
            idx = saved + i
            fname = f"{args.exp}_{ckpt_stem}_{step_tag}_sample_{idx:06d}.png"
            out_path = os.path.join(out_dir, fname)
            save_image(samples_01[i], out_path)

        if args.save_grid and (not preview_saved) and preview_batches_done < args.preview_batches:
            preview_cache.append(samples_01.detach().cpu())
            preview_batches_done += 1

            if preview_batches_done == args.preview_batches:
                all_imgs = torch.cat(preview_cache, dim=0)
                if args.preview_max is not None and all_imgs.shape[0] > args.preview_max:
                    all_imgs = all_imgs[: args.preview_max]

                n = all_imgs.shape[0]
                nrow = int(math.sqrt(n))
                nrow = max(1, min(nrow, 16))

                grid = make_grid(all_imgs, nrow=nrow)
                grid_path = os.path.join(
                    args.outdir,
                    args.exp,
                    f"{args.exp}_{ckpt_stem}_{step_tag}_preview_{args.sampler}_passes{passes}_batches{args.preview_batches}_n{n}.png",
                )
                save_image(grid, grid_path)
                preview_saved = True
                print(f"[OK] Saved preview grid: {grid_path}")

        saved += bs
        print(f"[OK] Saved {saved}/{total_samples} samples to {out_dir}")

    if args.save_grid and (not preview_saved) and preview_cache:
        all_imgs = torch.cat(preview_cache, dim=0)
        if args.preview_max is not None and all_imgs.shape[0] > args.preview_max:
            all_imgs = all_imgs[: args.preview_max]
        n = all_imgs.shape[0]
        nrow = int(math.sqrt(n))
        nrow = max(1, min(nrow, 16))

        grid = make_grid(all_imgs, nrow=nrow)
        grid_path = os.path.join(
            args.outdir,
            args.exp,
            f"{args.exp}_{ckpt_stem}_{step_tag}_preview_{args.sampler}_passes{passes}_batches{preview_batches_done}_n{n}.png",
        )
        save_image(grid, grid_path)
        print(f"[OK] Saved preview grid: {grid_path}")

    del lit_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", type=str, required=True, help="Exp number (e.g., exp7)")

    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--ckpt", type=str, help="Path to a single .ckpt")
    grp.add_argument("--ckpt_dir", type=str, help="Folder containing multiple .ckpt files")

    ap.add_argument("--ckpt_step", type=int, default=None, help="Override ckpt step (only for --ckpt single mode).")

    ap.add_argument("--outdir", type=str, default="./gen_out")
    ap.add_argument("--n", type=int, default=16, help="(Deprecated) batch size and total if --num_samples not set")
    ap.add_argument("--num_samples", type=int, default=None, help="Total number of samples to generate (per checkpoint)")
    ap.add_argument("--batch_size", type=int, default=None, help="Batch size per sampling pass")

    ap.add_argument("--save_grid", action="store_true", help="Save a preview grid (uses --preview_batches batches)")
    ap.add_argument("--preview_batches", type=int, default=10, help="How many batches to include in preview grid")
    ap.add_argument("--preview_max", type=int, default=None, help="Optional cap on number of preview images")

    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--channels", type=int, default=3)

    ap.add_argument("--num_timesteps", type=int, default=1000)
    ap.add_argument("--schedule", type=str, default="linear", choices=["linear", "cosine"])
    ap.add_argument("--beta_start", type=float, default=1e-4)
    ap.add_argument("--beta_end", type=float, default=2e-2)
    ap.add_argument("--cosine_s", type=float, default=0.008)

    ap.add_argument("--sampler", type=str, default="ddim", choices=["ddim", "ddpm"])
    ap.add_argument("--sample_steps", type=int, default=50, help="Number of passes/model calls (used by DDIM)")
    ap.add_argument("--eta", type=float, default=0.0, help="DDIM stochasticity. 0=deterministic")
    ap.add_argument("--clip_denoised", action="store_true", help="Clamp predicted x0 to [-1,1] during sampling")

    ap.add_argument("--class_id", type=int, default=None, help="If trained conditional, choose class id")
    ap.add_argument("--guidance_scale", type=float, default=1.0, help=">1 enables CFG (only if class_id set)")

    # NEW: choose attention vs no-attention LightningModule class
    ap.add_argument(
        "--attention",
        type=str2bool,
        default=True,
        help="true: use DiffusionUNetLit (with attention). false: use DiffusionUNetLitNoAttn.",
    )

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