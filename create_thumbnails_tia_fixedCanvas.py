from __future__ import annotations

import os
import csv
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from scipy.ndimage import (
    binary_fill_holes,
    binary_dilation as ndi_binary_dilation,
    binary_closing as ndi_binary_closing,
)
from skimage.morphology import disk
from skimage.measure import label, regionprops

from tiatoolbox.wsicore.wsireader import WSIReader


# -------------------------
# Thumbnail utilities
# -------------------------

def _thumb_to_rgb_uint8_np(x) -> np.ndarray:
    """
    Accepts PIL.Image or np.ndarray from tiatoolbox slide_thumbnail().
    Returns RGB uint8 ndarray (H, W, 3).
    """
    if isinstance(x, Image.Image):
        if x.mode != "RGB":
            x = x.convert("RGB")
        return np.asarray(x, dtype=np.uint8)

    if isinstance(x, np.ndarray):
        arr = x
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).round().astype(np.uint8)
        else:
            arr = arr.astype(np.uint8, copy=False)

        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)

        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        elif arr.ndim == 3 and arr.shape[2] == 3:
            pass
        else:
            raise ValueError(f"Unexpected thumbnail shape: {arr.shape}")

        return arr

    raise TypeError(f"Unsupported thumbnail type: {type(x)}")


def get_thumbnail(reader: WSIReader, thumbnail_mpp: float) -> np.ndarray:
    """Thumbnail at chosen MPP."""
    thumb_obj = reader.slide_thumbnail(resolution=float(thumbnail_mpp), units="mpp")
    return _thumb_to_rgb_uint8_np(thumb_obj)


# -------------------------
# Tissue detection utilities
# -------------------------

def make_masks_for_components_and_final(
    rgb_img_np: np.ndarray,
    merge_radius_px: int = 12,
):
    """
    - component_mask: Otsu + fill holes + optional closing to bridge thin gaps
    - final_mask    : Otsu + dilation + fill holes for nicer borders
    """
    gray = rgb2gray(rgb_img_np).astype(np.float32)
    T = threshold_otsu(gray)
    base = gray < T  # True = tissue

    component_mask = binary_fill_holes(base)
    if merge_radius_px and merge_radius_px > 0:
        se_merge = disk(merge_radius_px).astype(bool)
        component_mask = ndi_binary_closing(component_mask, structure=se_merge, iterations=1)

    se_final = disk(5).astype(bool)
    dil = ndi_binary_dilation(base, structure=se_final, iterations=1)
    final_mask = binary_fill_holes(dil)

    return component_mask, final_mask


# -------------------------
# CSV helpers
# -------------------------

def read_paths_from_csv(csv_path: str, col: str = "path"):
    paths = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if col not in (reader.fieldnames or []):
            raise ValueError(f"CSV must contain column '{col}'. Found: {reader.fieldnames}")
        for row in reader:
            p = (row[col] or "").strip()
            if p:
                paths.append(p)
    return paths


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# -------------------------
# Component selection logic
# -------------------------

def selected_component_bboxes_from_thumb(
    thumb_rgb: np.ndarray,
    pad: int = 20,
    min_area_ratio: float = 0.01,
    merge_radius_px: int = 20,
    use_8_connectivity: bool = True,
    split_area_frac: float = 0.25,
    density_single_thresh: float = 0.5,
    small_component_policy: str = "union",
):
    """
    Returns:
      selected_bboxes: list of (rmin, rmax, cmin, cmax, area_px, selection_type)
      final_mask
      comp_mask
    """
    if small_component_policy not in ("union", "largest"):
        raise ValueError("small_component_policy must be 'union' or 'largest'")

    H, W = thumb_rgb.shape[:2]

    comp_mask, final_mask = make_masks_for_components_and_final(
        thumb_rgb,
        merge_radius_px=merge_radius_px,
    )

    connectivity = 2 if use_8_connectivity else 1
    labeled = label(comp_mask, connectivity=connectivity)
    props = regionprops(labeled)

    if not props:
        return [(0, H, 0, W, 0, "no_tissue_fallback")], final_mask, comp_mask

    total_area = int(comp_mask.sum())
    min_area = max(1, int(total_area * min_area_ratio))
    kept = [p for p in props if p.area >= min_area]
    if not kept:
        kept = [max(props, key=lambda p: p.area)]
    kept.sort(key=lambda p: p.area, reverse=True)

    sum_area = float(sum(p.area for p in kept))

    rmin_all = min(p.bbox[0] for p in kept)
    cmin_all = min(p.bbox[1] for p in kept)
    rmax_all = max(p.bbox[2] for p in kept)
    cmax_all = max(p.bbox[3] for p in kept)

    union_h = rmax_all - rmin_all
    union_w = cmax_all - cmin_all
    union_area = max(1, union_h * union_w)
    density = sum_area / union_area

    area_fracs = [p.area / sum_area for p in kept]
    big_idxs = [i for i, f in enumerate(area_fracs) if f >= split_area_frac]

    selected = []

    if len(big_idxs) >= 2:
        for i in big_idxs:
            p = kept[i]
            r0, c0, r1, c1 = p.bbox
            rmin = max(0, r0 - pad)
            rmax = min(H, r1 + pad)
            cmin = max(0, c0 - pad)
            cmax = min(W, c1 + pad)
            selected.append((rmin, rmax, cmin, cmax, int(p.area), "split_big_component"))
        return selected, final_mask, comp_mask

    if small_component_policy == "largest" and density < density_single_thresh:
        p = kept[0]
        r0, c0, r1, c1 = p.bbox
        rmin = max(0, r0 - pad)
        rmax = min(H, r1 + pad)
        cmin = max(0, c0 - pad)
        cmax = min(W, c1 + pad)
        selected.append((rmin, rmax, cmin, cmax, int(p.area), "largest_component"))
        return selected, final_mask, comp_mask

    rmin = max(0, rmin_all - pad)
    rmax = min(H, rmax_all + pad)
    cmin = max(0, cmin_all - pad)
    cmax = min(W, cmax_all + pad)
    selected.append((rmin, rmax, cmin, cmax, int(sum_area), "union"))
    return selected, final_mask, comp_mask


# -------------------------
# Analysis
# -------------------------

def analyze_crop_sizes(
    csv_path: str,
    output_dir: str,
    col: str = "path",
    thumbnail_mpp: float = 8.0,
    pad: int = 20,
    min_area_ratio: float = 0.01,
    merge_radius_px: int = 20,
    use_8_connectivity: bool = True,
    split_area_frac: float = 0.25,
    density_single_thresh: float = 0.5,
    small_component_policy: str = "union",
    histogram_bins: int = 50,
):
    ensure_dir(output_dir)

    wsi_paths = read_paths_from_csv(csv_path, col=col)

    rows = []
    error_rows = []

    for idx, wsi_path in enumerate(wsi_paths, start=1):
        print(f"[{idx}/{len(wsi_paths)}] {wsi_path}")

        if not os.path.exists(wsi_path):
            error_rows.append({
                "wsi_path": wsi_path,
                "error": "missing_file",
            })
            print("  -> missing file")
            continue

        try:
            reader = WSIReader.open(wsi_path)
            thumb_rgb = get_thumbnail(reader, thumbnail_mpp=thumbnail_mpp)
            thumb_h, thumb_w = thumb_rgb.shape[:2]

            selected_bboxes, final_mask, comp_mask = selected_component_bboxes_from_thumb(
                thumb_rgb=thumb_rgb,
                pad=pad,
                min_area_ratio=min_area_ratio,
                merge_radius_px=merge_radius_px,
                use_8_connectivity=use_8_connectivity,
                split_area_frac=split_area_frac,
                density_single_thresh=density_single_thresh,
                small_component_policy=small_component_policy,
            )

            n_selected = len(selected_bboxes)
            total_tissue_px = int(comp_mask.sum())

            for part_idx, (rmin, rmax, cmin, cmax, area_px, selection_type) in enumerate(selected_bboxes, start=1):
                crop_h = int(rmax - rmin)
                crop_w = int(cmax - cmin)
                crop_long_side = int(max(crop_h, crop_w))
                crop_short_side = int(min(crop_h, crop_w))
                crop_box_area = int(crop_h * crop_w)
                crop_aspect_ratio = float(crop_w / crop_h) if crop_h > 0 else np.nan
                tissue_density_in_box = float(area_px / crop_box_area) if crop_box_area > 0 else np.nan

                rows.append({
                    "wsi_path": wsi_path,
                    "slide_name": os.path.splitext(os.path.basename(wsi_path))[0],
                    "thumbnail_mpp": thumbnail_mpp,
                    "thumbnail_h": int(thumb_h),
                    "thumbnail_w": int(thumb_w),
                    "thumbnail_long_side": int(max(thumb_h, thumb_w)),
                    "thumbnail_short_side": int(min(thumb_h, thumb_w)),
                    "total_tissue_px": total_tissue_px,
                    "n_selected_crops_for_wsi": n_selected,
                    "part_idx": part_idx,
                    "selection_type": selection_type,
                    "bbox_rmin": int(rmin),
                    "bbox_rmax": int(rmax),
                    "bbox_cmin": int(cmin),
                    "bbox_cmax": int(cmax),
                    "crop_h": crop_h,
                    "crop_w": crop_w,
                    "crop_long_side": crop_long_side,
                    "crop_short_side": crop_short_side,
                    "crop_box_area": crop_box_area,
                    "selected_area_px": int(area_px),
                    "tissue_density_in_box": tissue_density_in_box,
                    "crop_aspect_ratio_w_over_h": crop_aspect_ratio,
                })

        except Exception as e:
            error_rows.append({
                "wsi_path": wsi_path,
                "error": str(e),
            })
            print(f"  -> ERROR: {e}")

    if not rows:
        raise RuntimeError("No valid crop measurements were collected.")

    df = pd.DataFrame(rows)
    df_errors = pd.DataFrame(error_rows)

    csv_all_path = os.path.join(output_dir, "crop_size_analysis_all_rows.csv")
    df.to_csv(csv_all_path, index=False)

    if len(df_errors) > 0:
        csv_err_path = os.path.join(output_dir, "crop_size_analysis_errors.csv")
        df_errors.to_csv(csv_err_path, index=False)
    else:
        csv_err_path = None

    # Summary table
    metrics = {
        "crop_long_side": df["crop_long_side"].dropna().values,
        "crop_h": df["crop_h"].dropna().values,
        "crop_w": df["crop_w"].dropna().values,
        "crop_box_area": df["crop_box_area"].dropna().values,
        "tissue_density_in_box": df["tissue_density_in_box"].dropna().values,
    }

    summary_rows = []
    percentiles = [50, 75, 90, 95, 99, 100]

    for metric_name, arr in metrics.items():
        arr = np.asarray(arr)
        if arr.size == 0:
            continue

        row = {
            "metric": metric_name,
            "count": int(arr.size),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
        }
        for p in percentiles:
            row[f"p{p}"] = float(np.percentile(arr, p))
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    csv_summary_path = os.path.join(output_dir, "crop_size_analysis_summary.csv")
    df_summary.to_csv(csv_summary_path, index=False)

    # Histograms
    hist_path = os.path.join(output_dir, "crop_size_histograms.png")

    fig = plt.figure(figsize=(16, 10))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.hist(df["crop_long_side"], bins=histogram_bins)
    ax1.set_title("Crop longest side")
    ax1.set_xlabel("Pixels")
    ax1.set_ylabel("Count")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.hist(df["crop_h"], bins=histogram_bins)
    ax2.set_title("Crop height")
    ax2.set_xlabel("Pixels")
    ax2.set_ylabel("Count")

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.hist(df["crop_w"], bins=histogram_bins)
    ax3.set_title("Crop width")
    ax3.set_xlabel("Pixels")
    ax3.set_ylabel("Count")

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.hist(df["tissue_density_in_box"].dropna(), bins=histogram_bins)
    ax4.set_title("Tissue density in selected crop box")
    ax4.set_xlabel("Density")
    ax4.set_ylabel("Count")

    fig.suptitle("Selected crop dimensionality distribution", fontsize=14)
    fig.tight_layout()
    fig.savefig(hist_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # One extra histogram focused only on longest side, bigger and cleaner
    hist_long_side_path = os.path.join(output_dir, "crop_long_side_histogram.png")
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(df["crop_long_side"], bins=histogram_bins)
    ax.set_title("Distribution of selected crop longest side")
    ax.set_xlabel("Longest side (pixels)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(hist_long_side_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\nAnalysis completed.")
    print(f"All rows CSV:      {csv_all_path}")
    print(f"Summary CSV:       {csv_summary_path}")
    print(f"Histograms PNG:    {hist_path}")
    print(f"Long-side PNG:     {hist_long_side_path}")
    if csv_err_path is not None:
        print(f"Errors CSV:        {csv_err_path}")

    # Also print a few key numbers
    long_side = df["crop_long_side"].values
    print("\nKey percentiles for crop_long_side:")
    for p in [50, 75, 90, 95, 99, 100]:
        print(f"  p{p}: {np.percentile(long_side, p):.2f}")


# -------------------------
# Entry point
# -------------------------

if __name__ == "__main__":
    csv_path = "/storage/DSH/projects/iaso/diffusion_wholewsi/code/preprocessing/tcga_brca_histoqc_cleanList.csv"
    output_dir = "/storage/DSH/projects/iaso/data/BRCA_crop_size_analysis_thumbmpp8"

    analyze_crop_sizes(
        csv_path=csv_path,
        output_dir=output_dir,
        col="path",
        thumbnail_mpp=8.0,
        pad=20,
        min_area_ratio=0.01,
        merge_radius_px=20,
        use_8_connectivity=True,
        split_area_frac=0.25,
        density_single_thresh=0.5,
        small_component_policy="union",   # or "largest"
        histogram_bins=50,
    )