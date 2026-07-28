# generation/compare_fid.py
import argparse
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import torch

try:
    from pytorch_fid import fid_score, inception
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "pytorch-fid is required. Install it (e.g., `pip install pytorch-fid`) "
        "and ensure it is available in your environment."
    ) from exc

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def list_images(directory: str) -> list[str]:
    paths: list[str] = []
    d = Path(directory)
    if not d.exists():
        return []
    for p in d.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            paths.append(str(p))
    return sorted(paths)


def detect_exp_name(path_str: str, override: str | None = None) -> str:
    if override:
        return override
    m = re.search(r"(exp[_-]?\d+)", path_str, flags=re.IGNORECASE)
    if m:
        return m.group(1).replace("-", "_").lower()
    return "exp"


def infer_step_from_string(s: str) -> int | None:
    """
    Infer *training checkpoint step* from a string.

    Must:
      - match your ckpt token: "-stepstep=10000"
      - match common tokens: "step=10000", "global_step=10000", "_step10000_"
      - NOT match diffusion denoising token: "steps1000" (plural)
    """
    txt = str(s)

    patterns = [
        # Your exact convention: "...-stepstep=10000.ckpt"
        r"(?:^|[^a-zA-Z0-9])stepstep\s*=\s*([0-9]+)(?:$|[^0-9])",

        # Common: "...step=10000..."  (avoid "steps1000")
        r"(?:^|[^a-zA-Z0-9])step(?!s)\s*=\s*([0-9]+)(?:$|[^0-9])",

        # Common: "..._step10000_..." or ".../step10000/..." or "...-step10000..."
        r"(?:^|[^a-zA-Z0-9])step(?!s)[_\-\/]?([0-9]+)(?:$|[^0-9])",

        # Lightning: global_step=10000 / global_step10000
        r"(?:^|[^a-zA-Z0-9])global[_\-]?step\s*=?\s*([0-9]+)(?:$|[^0-9])",
    ]

    for pat in patterns:
        m = re.search(pat, txt, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def infer_step_from_model_dir(model_dir: Path) -> int | None:
    """
    Prefer training step from ckpt filename if present.
    Falls back to directory name.
    """
    # 1) directory name
    step = infer_step_from_string(model_dir.name)
    if step is not None:
        return step

    # 2) ckpt file names inside
    ckpts = sorted(model_dir.rglob("*.ckpt"))
    for ck in ckpts:
        step = infer_step_from_string(ck.name)
        if step is not None:
            return step

    # 3) full path fallback
    return infer_step_from_string(str(model_dir))


def parse_sizes(s: str | None) -> list[int]:
    if not s:
        return []
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


# -----------------------------
# Parse experiment/model characteristics from checkpoint/folder name
# Template example:
# "20260218_093216_diffusionUNet_exp8_lr0.0001_bs32_wd0.0_dim64_steps1000_linear"
# -----------------------------
def _num_token_to_float(tok: str) -> float:
    return float(tok.replace("E", "e"))


def parse_model_characteristics(s: str) -> dict:
    """
    Extract:
      - exp
      - lr
      - bs
      - wd
      - denoise_steps  (from plural "stepsXXXX" in your template)
    """
    txt = str(s)
    out: dict = {}

    # exp8 / exp_8 / exp-8
    m = re.search(r"(exp[_-]?\d+)", txt, flags=re.IGNORECASE)
    if m:
        out["exp"] = m.group(1).replace("-", "_").lower()

    # lr0.0001 / lr=1e-4 / lr_1e-4
    m = re.search(r"(?:^|[^a-zA-Z0-9])lr(?:=|_)?([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)", txt)
    if m:
        try:
            out["lr"] = _num_token_to_float(m.group(1))
        except Exception:
            pass

    # bs32 / bs=32 / batch32 / batch=32
    m = re.search(r"(?:^|[^a-zA-Z0-9])(bs|batch)(?:=|_)?(\d+)", txt, flags=re.IGNORECASE)
    if m:
        out["bs"] = int(m.group(2))

    # wd0.0 / wd=0.01 / weightdecay0.01
    m = re.search(
        r"(?:^|[^a-zA-Z0-9])(wd|weightdecay)(?:=|_)?([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)",
        txt,
        flags=re.IGNORECASE,
    )
    if m:
        try:
            out["wd"] = _num_token_to_float(m.group(2))
        except Exception:
            pass

    # IMPORTANT: denoising steps token is "steps1000" (plural).
    m = re.search(r"(?:^|[^a-zA-Z0-9])steps(?:=|_)?(\d+)(?:$|[^0-9])", txt, flags=re.IGNORECASE)
    if m:
        out["denoise_steps"] = int(m.group(1))

    return out


def build_plot_title(fallback_exp: str, characteristics: dict, suffix: str) -> str:
    exp = characteristics.get("exp", fallback_exp)

    parts = [str(exp)]
    if "lr" in characteristics:
        parts.append(f"lr={characteristics['lr']:.4g}")
    if "bs" in characteristics:
        parts.append(f"bs={int(characteristics['bs'])}")
    if "wd" in characteristics:
        parts.append(f"wd={characteristics['wd']:.4g}")
    if "denoise_steps" in characteristics:
        parts.append(f"denoise={int(characteristics['denoise_steps'])}")

    if len(parts) == 1:
        head = parts[0]
    else:
        head = f"{parts[0]} | " + " ".join(parts[1:])

    return f"{head}: {suffix}"


def step_label(step: int | None) -> str:
    return "best" if step is None else str(step)


def compute_activations(
    files: list[str],
    batch_size: int,
    device: torch.device,
    dims: int,
    num_workers: int,
) -> np.ndarray:
    """
    Returns Inception activations (N x dims) for the given image files using pytorch-fid.

    NOTE: pytorch-fid expects images in a batch to share the same HxW.
    """
    block_idx = inception.InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = inception.InceptionV3([block_idx]).to(device)
    model.eval()

    acts = fid_score.get_activations(
        files,
        model,
        batch_size=batch_size,
        dims=dims,
        device=device,
        num_workers=num_workers,
    )
    return acts


def stats_from_activations(activations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma


def fid_from_activations(a1: np.ndarray, a2: np.ndarray) -> float:
    mu1, sigma1 = stats_from_activations(a1)
    mu2, sigma2 = stats_from_activations(a2)
    return float(fid_score.calculate_frechet_distance(mu1, sigma1, mu2, sigma2))


def compute_real_split_baseline(
    real_files: list[str],
    size: int,
    repeats: int,
    rng: random.Random,
    batch_size: int,
    device: torch.device,
    dims: int,
    num_workers: int,
) -> dict:
    """
    Baseline = FID(A, B) where A and B are disjoint samples from REAL set.
    Needs 2*size real images available.
    """
    if 2 * size > len(real_files):
        raise SystemExit(
            f"Baseline needs 2*size real images. Requested size={size}, "
            f"but only {len(real_files)} real images available."
        )

    vals: list[float] = []
    for _ in range(repeats):
        subset = rng.sample(real_files, 2 * size)
        a = subset[:size]
        b = subset[size:]

        a_acts = compute_activations(a, batch_size, device, dims, num_workers)
        b_acts = compute_activations(b, batch_size, device, dims, num_workers)

        vals.append(fid_from_activations(a_acts, b_acts))

    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return {"mean": mean, "std": std, "values": vals}


def plot_fid_distributions(
    real_acts: np.ndarray,
    fake_acts: np.ndarray,
    fid_value: float,
    outdir: str,
    prefix: str,
    max_points: int = 8000,
    baseline_mean: float | None = None,
    baseline_std: float | None = None,
    baseline_size: int | None = None,
) -> str:
    """
    2-panel diagnostic plot:
      (1) PCA 2D scatter of Inception features with covariance ellipses + mean markers
      (2) Overlaid histogram of PC1 (real vs fake)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    os.makedirs(outdir, exist_ok=True)

    Xr = real_acts.astype(np.float64, copy=False)
    Xf = fake_acts.astype(np.float64, copy=False)

    X = np.vstack([Xr, Xf])
    X_mean = X.mean(axis=0, keepdims=True)
    Xc = X - X_mean

    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    W2 = Vt[:2].T
    Z = Xc @ W2

    zr = Z[: len(Xr)]
    zf = Z[len(Xr):]

    rng = np.random.default_rng(123)

    def subsample(Zin: np.ndarray, n: int) -> np.ndarray:
        if Zin.shape[0] <= n:
            return Zin
        idx = rng.choice(Zin.shape[0], size=n, replace=False)
        return Zin[idx]

    zr_sc = subsample(zr, max_points)
    zf_sc = subsample(zf, max_points)

    def add_cov_ellipse(ax, Z2: np.ndarray, n_std: float = 2.0) -> np.ndarray:
        mu = Z2.mean(axis=0)
        cov = np.cov(Z2.T)

        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]

        width, height = 2 * n_std * np.sqrt(np.maximum(vals, 1e-12))
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

        ell = Ellipse(xy=mu, width=width, height=height, angle=angle, fill=False, linewidth=2)
        ax.add_patch(ell)
        return mu

    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(zr_sc[:, 0], zr_sc[:, 1], s=6, alpha=0.25, label="Real")
    ax1.scatter(zf_sc[:, 0], zf_sc[:, 1], s=6, alpha=0.25, label="Fake")
    mu_r = add_cov_ellipse(ax1, zr, n_std=2.0)
    mu_f = add_cov_ellipse(ax1, zf, n_std=2.0)
    ax1.scatter([mu_r[0]], [mu_r[1]], s=70, marker="x")
    ax1.scatter([mu_f[0]], [mu_f[1]], s=70, marker="x")
    ax1.set_title("Inception features (PCA 2D) + cov ellipses")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")

    ax2 = fig.add_subplot(1, 2, 2)
    bins = min(80, max(20, int(np.sqrt(zr.shape[0] + zf.shape[0]))))
    ax2.hist(zr[:, 0], bins=bins, alpha=0.5, density=True, label="Real (PC1)")
    ax2.hist(zf[:, 0], bins=bins, alpha=0.5, density=True, label="Fake (PC1)")
    ax2.set_title("Two distributions (PC1)")
    ax2.set_xlabel("PC1 value")
    ax2.set_ylabel("Density")

    base_str = ""
    if baseline_mean is not None:
        if baseline_std is None:
            base_str = f"  |  Baseline(real split)={baseline_mean:.4f}"
        else:
            base_str = f"  |  Baseline(real split)={baseline_mean:.4f}±{baseline_std:.4f}"
        if baseline_size is not None:
            base_str += f" (n={baseline_size})"

    fig.suptitle(f"{prefix}  |  FID(real,fake)={fid_value:.4f}{base_str}", fontsize=13)

    if baseline_mean is not None:
        txt = f"FID(real,fake) = {fid_value:.4f}\nBaseline(real split) = {baseline_mean:.4f}"
        if baseline_std is not None:
            txt += f" ± {baseline_std:.4f}"
        if baseline_size is not None:
            txt += f"\nBaseline n = {baseline_size}"
        ax2.text(
            0.02, 0.98, txt,
            transform=ax2.transAxes,
            ha="left", va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="0.8"),
        )

    fig.text(
        0.5,
        0.02,
        "Note: PCA is illustrative; FID uses full mean+cov in the original feature space.",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    # shared legend outside the figure (right side)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2

    seen = set()
    uniq = []
    for h, l in zip(handles, labels):
        if l not in seen:
            uniq.append((h, l))
            seen.add(l)
    if uniq:
        handles, labels = zip(*uniq)
        fig.legend(
            handles, labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            fontsize=9,
        )

    outpath = os.path.join(outdir, f"{prefix}_distributions.png")
    plt.tight_layout(rect=[0, 0.05, 0.82, 0.93])
    plt.savefig(outpath, dpi=170)
    plt.close(fig)
    return outpath


def is_multi_model_root(fake_dir: str) -> tuple[bool, list[Path]]:
    """
    Detect if fake_dir is a parent directory containing multiple model subfolders.
    We consider "multi-model" if it has at least one immediate subdirectory
    containing images (recursively).

    This intentionally ignores root-level images so FID only uses images inside model subfolders.
    Returns (is_multi, subdirs).
    """
    root = Path(fake_dir)
    if not root.is_dir():
        return (False, [])

    children = [p for p in root.iterdir() if p.is_dir()]
    child_with_imgs = [c for c in children if list_images(str(c))]
    if child_with_imgs:
        return (True, sorted(child_with_imgs, key=lambda p: p.name))
    return (False, [])


def fid_single_mode(
    real_files: list[str],
    fake_files: list[str],
    args,
    rng: random.Random,
    device: torch.device,
    prefix: str,
    outdir: str,
) -> dict:
    n_eval = min(len(real_files), len(fake_files))
    n_base = min(n_eval, len(real_files) // 2)

    real_acts = compute_activations(real_files[:n_eval], args.batch_size, device, args.dims, args.num_workers)
    fake_acts = compute_activations(fake_files[:n_eval], args.batch_size, device, args.dims, args.num_workers)
    fid = fid_from_activations(real_acts, fake_acts)

    baseline = compute_real_split_baseline(
        real_files=real_files,
        size=n_base,
        repeats=args.baseline_repeats,
        rng=rng,
        batch_size=args.batch_size,
        device=device,
        dims=args.dims,
        num_workers=args.num_workers,
    )

    results: dict = {
        "fid": fid,
        "n_eval": n_eval,
        "baseline_real_split": {"size": n_base, **baseline},
    }

    if args.plot:
        viz_path = plot_fid_distributions(
            real_acts,
            fake_acts,
            fid_value=fid,
            outdir=outdir,
            prefix=prefix,
            max_points=args.plot_max_points,
            baseline_mean=baseline["mean"],
            baseline_std=baseline["std"],
            baseline_size=n_base,
        )
        results["fid_plot"] = os.path.basename(viz_path)
        print(f"[OK] Saved plot: {viz_path}")

    return results


def fid_curve_mode(
    real_files: list[str],
    fake_files: list[str],
    args,
    rng: random.Random,
    device: torch.device,
    prefix: str,
    outdir: str,
    sample_sizes: list[int],
    plot_title: str | None = None,
    legend_label: str | None = None,
) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve: list[dict] = []
    dist_plots: list[str] = []

    for size in sample_sizes:
        if size > len(real_files) or size > len(fake_files):
            raise SystemExit(
                f"Requested sample size {size} exceeds available images: "
                f"real={len(real_files)}, fake={len(fake_files)}"
            )
        if 2 * size > len(real_files):
            raise SystemExit(
                f"Baseline needs 2*size real images for size={size}, but real={len(real_files)}."
            )

        baseline = compute_real_split_baseline(
            real_files=real_files,
            size=size,
            repeats=args.baseline_repeats,
            rng=rng,
            batch_size=args.batch_size,
            device=device,
            dims=args.dims,
            num_workers=args.num_workers,
        )

        fids: list[float] = []
        for rep in range(args.num_repeats):
            real_subset = rng.sample(real_files, size)
            fake_subset = rng.sample(fake_files, size)

            real_acts = compute_activations(real_subset, args.batch_size, device, args.dims, args.num_workers)
            fake_acts = compute_activations(fake_subset, args.batch_size, device, args.dims, args.num_workers)

            fid = fid_from_activations(real_acts, fake_acts)
            fids.append(fid)

            if args.plot and (args.plot_every_repeat or rep == 0):
                plot_prefix = f"{prefix}_n{size}" + (f"_rep{rep+1}" if args.plot_every_repeat else "")
                viz_path = plot_fid_distributions(
                    real_acts,
                    fake_acts,
                    fid_value=fid,
                    outdir=outdir,
                    prefix=plot_prefix,
                    max_points=args.plot_max_points,
                    baseline_mean=baseline["mean"],
                    baseline_std=baseline["std"],
                    baseline_size=size,
                )
                dist_plots.append(os.path.basename(viz_path))
                print(f"[OK] Saved plot: {viz_path}")

        curve.append(
            {
                "size": size,
                "model_mean": float(np.mean(fids)),
                "model_std": float(np.std(fids, ddof=1)) if len(fids) > 1 else 0.0,
                "model_values": fids,
                "baseline_mean": baseline["mean"],
                "baseline_std": baseline["std"],
                "baseline_values": baseline["values"],
            }
        )

        print(
            f"[OK] size={size} model_fid={curve[-1]['model_mean']:.4f}±{curve[-1]['model_std']:.4f} "
            f"baseline={curve[-1]['baseline_mean']:.4f}±{curve[-1]['baseline_std']:.4f}"
        )

    results: dict = {"curve": curve}
    if dist_plots:
        results["distribution_plots"] = dist_plots

    sizes = [c["size"] for c in curve]
    model_means = [c["model_mean"] for c in curve]
    model_stds = [c["model_std"] for c in curve]
    base_means = [c["baseline_mean"] for c in curve]
    base_stds = [c["baseline_std"] for c in curve]

    plt.figure(figsize=(7, 4.5))
    lbl = legend_label or "Model: real vs fake"
    plt.errorbar(sizes, model_means, yerr=model_stds, marker="o", linewidth=2, capsize=4, label=lbl)
    plt.errorbar(
        sizes,
        base_means,
        yerr=base_stds,
        marker="o",
        linewidth=2,
        capsize=4,
        label="Baseline: real split-half",
    )

    plt.title(plot_title or f"{prefix}: FID vs Sample Size (with baseline)")
    plt.xlabel("Samples per set")
    plt.ylabel("FID (lower is better)")
    plt.grid(True, alpha=0.3)

    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9, frameon=True)

    plot_path = os.path.join(outdir, f"{prefix}_fid_curve.png")
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(plot_path, dpi=150)
    plt.close()
    results["fid_curve_plot"] = os.path.basename(plot_path)

    print(f"[OK] Saved: {plot_path}")
    return results


def plot_group_fid_vs_step(
    entries: list[dict],
    outdir: str,
    prefix: str,
    y_key: str,
    y_label: str,
    plot_title: str | None = None,
) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)

    def sort_key(e):
        return (-1 if e.get("step") is None else e.get("step"))

    ordered = sorted(entries, key=sort_key)
    x_pos = list(range(len(ordered)))
    x_labels = [step_label(e.get("step")) for e in ordered]
    y_vals = [float(e[y_key]) for e in ordered]

    plt.figure(figsize=(max(7, 0.8 * len(ordered)), 4.5))
    plt.plot(x_pos, y_vals, marker="o", linewidth=2)
    plt.xticks(x_pos, x_labels, rotation=45, ha="right")
    plt.title(plot_title or f"{prefix}: {y_label} vs checkpoint step")
    plt.xlabel("Checkpoint step")
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.3)

    outpath = os.path.join(outdir, f"{prefix}_group_{y_label.replace(' ', '_')}_vs_step.png")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()
    return outpath


def plot_group_fid_curves(
    entries: list[dict],
    outdir: str,
    prefix: str,
    plot_title: str | None = None,
) -> str:
    """
    Group curve plot:
      - one FID-vs-sample-size curve per model
      - labels are ONLY training step: best, 1000, 2000, ...
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)

    all_sizes = sorted({int(pt["size"]) for e in entries for pt in e.get("curve", [])})
    if not all_sizes:
        raise SystemExit("[ERROR] Group curve plot requested but no curve data found.")

    baseline_by_size: dict[int, list[float]] = {s: [] for s in all_sizes}

    plt.figure(figsize=(8.5, 5.2))

    def sort_key(e):
        return (-1 if e.get("step") is None else e.get("step"))

    for e in sorted(entries, key=sort_key):
        curve = e.get("curve", [])
        if not curve:
            continue

        by_size = {int(pt["size"]): pt for pt in curve}
        sizes = [s for s in all_sizes if s in by_size]
        if not sizes:
            continue

        model_means = [float(by_size[s]["model_mean"]) for s in sizes]
        model_stds = [float(by_size[s]["model_std"]) for s in sizes]
        label = step_label(e.get("step"))

        plt.errorbar(
            sizes,
            model_means,
            yerr=model_stds,
            marker="o",
            linewidth=1.8,
            capsize=3,
            alpha=0.9,
            label=label,
        )

        for s in sizes:
            baseline_by_size[s].append(float(by_size[s]["baseline_mean"]))

    base_sizes = [s for s in all_sizes if baseline_by_size[s]]
    base_means = [float(np.mean(baseline_by_size[s])) for s in base_sizes]
    if base_sizes:
        plt.plot(
            base_sizes,
            base_means,
            marker="s",
            linestyle="--",
            linewidth=2.2,
            color="black",
            label="Baseline: real split-half",
        )

    plt.title(plot_title or f"{prefix}: FID Curves Across Models")
    plt.xlabel("Samples per set")
    plt.ylabel("FID (lower is better)")
    plt.grid(True, alpha=0.3)

    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=True)

    outpath = os.path.join(outdir, f"{prefix}_group_fid_curves_vs_samples.png")
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(outpath, dpi=170)
    plt.close()
    return outpath


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", type=str, required=True, help="Directory with real images")
    ap.add_argument("--fake_dir", type=str, required=True, help="Directory with fake images OR parent with subfolders per ckpt")
    ap.add_argument("--outdir", type=str, default="./fid_out")

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--dims", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--max_real", type=int, default=None)
    ap.add_argument("--max_fake", type=int, default=None)

    ap.add_argument("--sample_sizes", type=str, default=None)
    ap.add_argument("--num_repeats", type=int, default=3)

    ap.add_argument("--exp", type=str, default=None, help="Optional exp override (e.g., exp7)")
    ap.add_argument("--step", type=int, default=None, help="Optional step override (only used in single-folder mode)")
    ap.add_argument("--ckpt", type=str, default=None, help="Optional ckpt path; step can be inferred from it (single-folder mode)")

    ap.add_argument("--plot", action="store_true", help="Save distribution plots (PCA+PC1 hist) with FID highlighted")
    ap.add_argument("--plot_max_points", type=int, default=8000)
    ap.add_argument("--plot_every_repeat", action="store_true")

    ap.add_argument("--baseline_repeats", type=int, default=5)

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device)

    real_files = list_images(args.real_dir)
    if args.max_real:
        real_files = real_files[: args.max_real]
    if not real_files:
        raise SystemExit(f"No images found in real_dir: {args.real_dir}")

    rng = random.Random(args.seed)
    sample_sizes = parse_sizes(args.sample_sizes)

    exp_name = detect_exp_name(args.fake_dir, args.exp)

    multi, model_dirs = is_multi_model_root(args.fake_dir)

    # -------------------------
    # MULTI-MODEL
    # -------------------------
    if multi:
        print(f"[INFO] Detected multi-model folder: {args.fake_dir}")
        print(f"[INFO] Found {len(model_dirs)} model subfolders")

        group_entries: list[dict] = []

        base_meta = {
            "exp": exp_name,
            "real_dir": args.real_dir,
            "fake_root": args.fake_dir,
            "num_real": len(real_files),
            "dims": args.dims,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "seed": args.seed,
            "baseline_repeats": args.baseline_repeats,
            "mode": "curve" if sample_sizes else "single",
            "sample_sizes": sample_sizes if sample_sizes else None,
        }

        # Best-effort group plot title (grab lr/bs/wd/steps from first model dir name/path)
        group_chars = parse_model_characteristics(str(model_dirs[0]))
        if "exp" not in group_chars:
            group_chars["exp"] = exp_name

        for md in model_dirs:
            fake_files = list_images(str(md))
            if args.max_fake:
                fake_files = fake_files[: args.max_fake]
            if not fake_files:
                print(f"[WARN] No fake images in {md}, skipping")
                continue

            # IMPORTANT: infer training step from ckpt file if present
            ckpt_step = infer_step_from_model_dir(md)
            step_tag = step_label(ckpt_step)

            # Characteristics (lr/bs/wd/denoise steps) from dir path/name
            model_chars = parse_model_characteristics(str(md))
            if "exp" not in model_chars:
                model_chars["exp"] = exp_name

            model_name = md.name
            prefix = f"{exp_name}_{model_name}_step{step_tag}"

            results: dict = dict(base_meta)
            results.update(
                {
                    "model_dir": str(md),
                    "model_name": model_name,
                    "step": ckpt_step,
                    "step_label": step_tag,
                    "num_fake": len(fake_files),
                    "characteristics": model_chars,
                }
            )

            if not sample_sizes:
                sub = fid_single_mode(
                    real_files=real_files,
                    fake_files=fake_files,
                    args=args,
                    rng=rng,
                    device=device,
                    prefix=prefix,
                    outdir=args.outdir,
                )
                results.update(sub)

                out_json = os.path.join(args.outdir, f"{prefix}_fid.json")
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)

                group_entries.append(
                    {"step": ckpt_step, "label": step_tag, "fid": float(results["fid"]), "json": os.path.basename(out_json)}
                )

                print(f"[OK] {model_name} (step {step_tag}) FID: {results['fid']:.4f}")
                print(f"[OK] Saved: {out_json}")

            else:
                title = build_plot_title(exp_name, model_chars, "FID vs Sample Size (with baseline)")
                sub = fid_curve_mode(
                    real_files=real_files,
                    fake_files=fake_files,
                    args=args,
                    rng=rng,
                    device=device,
                    prefix=prefix,
                    outdir=args.outdir,
                    sample_sizes=sample_sizes,
                    plot_title=title,
                    legend_label=step_tag,  # only step label
                )
                results.update(sub)

                out_json = os.path.join(args.outdir, f"{prefix}_fid_curve.json")
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)

                last = results["curve"][-1]
                curve_y = float(last["model_mean"])

                curve_summary = [
                    {
                        "size": int(p["size"]),
                        "model_mean": float(p["model_mean"]),
                        "model_std": float(p["model_std"]),
                        "baseline_mean": float(p["baseline_mean"]),
                        "baseline_std": float(p["baseline_std"]),
                    }
                    for p in results["curve"]
                ]

                group_entries.append(
                    {
                        "step": ckpt_step,
                        "label": step_tag,
                        "curve_y": curve_y,
                        "curve_size": int(last["size"]),
                        "curve": curve_summary,
                        "json": os.path.basename(out_json),
                    }
                )

                print(f"[OK] {model_name} (step {step_tag}) curve@{last['size']}: {curve_y:.4f}")
                print(f"[OK] Saved: {out_json}")

        if not group_entries:
            raise SystemExit("[ERROR] No valid model subfolders with images found.")

        group_prefix = f"{exp_name}_group"
        group_json = os.path.join(args.outdir, f"{group_prefix}_fid_group.json")
        group_payload = dict(base_meta)
        group_payload["characteristics"] = group_chars
        group_payload["models"] = group_entries
        with open(group_json, "w", encoding="utf-8") as f:
            json.dump(group_payload, f, indent=2)
        print(f"[OK] Saved group summary: {group_json}")

        if not sample_sizes:
            group_title = build_plot_title(exp_name, group_chars, "FID vs checkpoint step")
            plot_path = plot_group_fid_vs_step(
                entries=group_entries,
                outdir=args.outdir,
                prefix=group_prefix,
                y_key="fid",
                y_label="FID",
                plot_title=group_title,
            )
            print(f"[OK] Saved group plot: {plot_path}")
        else:
            group_title = build_plot_title(exp_name, group_chars, "FID Curves Across Models")
            plot_path = plot_group_fid_curves(
                entries=group_entries,
                outdir=args.outdir,
                prefix=group_prefix,
                plot_title=group_title,
            )
            print(f"[OK] Saved group plot: {plot_path}")

        return

    # -------------------------
    # SINGLE-MODEL
    # -------------------------
    fake_files = list_images(args.fake_dir)
    if args.max_fake:
        fake_files = fake_files[: args.max_fake]
    if not fake_files:
        raise SystemExit(
            f"No images found in fake_dir: {args.fake_dir}\n"
            f"If you expected a multi-model run, pass the PARENT folder that contains subfolders per checkpoint."
        )

    ckpt_step = args.step
    if ckpt_step is None:
        ckpt_step = infer_step_from_string(args.fake_dir)
    if ckpt_step is None and args.ckpt is not None:
        ckpt_step = infer_step_from_string(args.ckpt)

    step_tag = step_label(ckpt_step)
    prefix = f"{exp_name}_step{step_tag}"

    single_chars = parse_model_characteristics(args.fake_dir)
    if "exp" not in single_chars:
        single_chars["exp"] = exp_name

    results: dict = {
        "exp": exp_name,
        "step": ckpt_step,
        "step_label": step_tag,
        "real_dir": args.real_dir,
        "fake_dir": args.fake_dir,
        "num_real": len(real_files),
        "num_fake": len(fake_files),
        "dims": args.dims,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "baseline_repeats": args.baseline_repeats,
        "characteristics": single_chars,
    }

    if not sample_sizes:
        sub = fid_single_mode(
            real_files=real_files,
            fake_files=fake_files,
            args=args,
            rng=rng,
            device=device,
            prefix=prefix,
            outdir=args.outdir,
        )
        results.update(sub)

        out_json = os.path.join(args.outdir, f"{prefix}_fid.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"[OK] FID: {results['fid']:.4f}")
        print(
            f"[OK] Baseline real-split (n={results['baseline_real_split']['size']}): "
            f"{results['baseline_real_split']['mean']:.4f} ± {results['baseline_real_split']['std']:.4f}"
        )
        print(f"[OK] Saved: {out_json}")
        return

    title = build_plot_title(exp_name, single_chars, "FID vs Sample Size (with baseline)")
    sub = fid_curve_mode(
        real_files=real_files,
        fake_files=fake_files,
        args=args,
        rng=rng,
        device=device,
        prefix=prefix,
        outdir=args.outdir,
        sample_sizes=sample_sizes,
        plot_title=title,
        legend_label=step_tag,
    )
    results.update(sub)

    out_json = os.path.join(args.outdir, f"{prefix}_fid_curve.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[OK] Saved: {out_json}")


if __name__ == "__main__":
    main()