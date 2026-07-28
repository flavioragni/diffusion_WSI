import os
import csv
import numpy as np
from PIL import Image

from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import disk
from skimage.measure import label, regionprops
from scipy.ndimage import (
    binary_fill_holes,
    binary_dilation as ndi_binary_dilation,
    binary_closing as ndi_binary_closing,
)

from tiatoolbox.wsicore.wsireader import WSIReader


# -------------------------
# Mask + resize utilities
# -------------------------

def make_masks_for_components_and_final(rgb_img_np: np.ndarray, merge_radius_px: int = 12):
    """
    Build two masks:
      - component_mask: Otsu + fill holes + optional closing (bridges small gaps)
      - final_mask    : Otsu + dilation (r=5) + fill holes (cleaner edges for saving)
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


def letterbox_to_512(
    rgb_np: np.ndarray,
    mask_np: np.ndarray,
    scale: float | None = None,
    target: int = 512,
):
    """
    Resize with aspect ratio then paste centered on a 512x512 canvas.
    If `scale` is provided, use it (shared per slide) to avoid per-component "zoom".
    Never upscales beyond 1.0.
    """
    h, w = rgb_np.shape[:2]
    if scale is None:
        scale = min(1.0, target / max(h, w))
    else:
        scale = min(scale, 1.0)

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    rgb_resized = Image.fromarray(rgb_np).resize((new_w, new_h), resample=Image.BILINEAR)
    mask_resized = Image.fromarray((mask_np.astype(np.uint8) * 255)).resize(
        (new_w, new_h), resample=Image.NEAREST
    )

    canvas_rgb = np.ones((target, target, 3), dtype=np.uint8) * 255
    canvas_mask = np.zeros((target, target), dtype=np.uint8)

    y0 = (target - new_h) // 2
    x0 = (target - new_w) // 2
    canvas_rgb[y0:y0 + new_h, x0:x0 + new_w, :] = np.asarray(rgb_resized)
    canvas_mask[y0:y0 + new_h, x0:x0 + new_w] = np.asarray(mask_resized)

    return Image.fromarray(canvas_rgb), Image.fromarray(canvas_mask)


# -------------------------
# Level selection utilities (avoid OOM)
# -------------------------

def _get_scalar_mpp(wsi):
    mpp = getattr(wsi.info, "mpp", None)
    if mpp is None:
        return None
    if isinstance(mpp, (list, tuple, np.ndarray)):
        return float(mpp[0])
    return float(mpp)


def choose_level_safe(wsi, target_mpp: float, max_pixels: int = 50_000_000):
    """
    Choose a pyramid level whose mpp >= target_mpp and closest to it,
    and ensure the level image is not too big (pixel budget).
    """
    level_dims = wsi.info.level_dimensions
    level_downs = wsi.info.level_downsamples
    num_levels = len(level_dims)

    base_mpp = _get_scalar_mpp(wsi)
    if base_mpp is None:
        idx = num_levels - 1
        return idx, None

    level_mpps = [base_mpp * float(d) for d in level_downs]

    candidates = [i for i, m in enumerate(level_mpps) if m >= target_mpp]
    if candidates:
        idx = min(candidates, key=lambda i: level_mpps[i] - target_mpp)
    else:
        idx = num_levels - 1

    while idx < num_levels - 1:
        w, h = level_dims[idx]
        if w * h <= max_pixels:
            break
        idx += 1

    return idx, level_mpps[idx]


# -------------------------
# Core processing
# -------------------------

def _clamp_bbox(rmin, rmax, cmin, cmax, H, W):
    rmin = max(0, rmin); cmin = max(0, cmin)
    rmax = min(H, rmax); cmax = min(W, cmax)
    if rmax <= rmin: rmax = min(H, rmin + 1)
    if cmax <= cmin: cmax = min(W, cmin + 1)
    return rmin, rmax, cmin, cmax


def process_wsi_to_512_multi(
    wsi_path: str,
    pad: int = 20,
    min_area_ratio: float = 0.01,
    merge_radius_px: int = 12,
    use_8_connectivity: bool = True,
    density_single_thresh: float = 0.75,
    target_mpp: float = 4.0,

    # NEW: split only if >=2 components are "big"
    split_area_frac: float = 0.33,   # "big component" if area >= 33% of total tissue area
    small_component_policy: str = "union",  # "union" or "largest"
):
    """
    Returns list of items:
      {'rgb_512','mask_512','bbox','area_px','chosen_level','approx_mpp'}

    Splitting rule:
      - Compute components and their areas.
      - If at least 2 components have area_frac >= split_area_frac -> SPLIT into those big components.
      - Else -> DO NOT split:
          * policy "union": return one crop covering UNION of kept components
          * policy "largest": return only the largest component crop
    """
    if small_component_policy not in ("union", "largest"):
        raise ValueError("small_component_policy must be 'union' or 'largest'")

    wsi = WSIReader.open(wsi_path)
    level_dims = wsi.info.level_dimensions

    chosen_level, approx_mpp = choose_level_safe(wsi, target_mpp=target_mpp)
    level_w, level_h = level_dims[chosen_level]

    lowres_rgba = wsi.read_region(location=(0, 0), level=chosen_level, size=(level_w, level_h))
    lowres_rgb = np.asarray(lowres_rgba)[..., :3]

    H, W = lowres_rgb.shape[:2]
    if (H == level_w) and (W == level_h) and (H != level_h):
        lowres_rgb = np.rot90(lowres_rgb, k=3)
        H, W = lowres_rgb.shape[:2]

    comp_mask, final_mask = make_masks_for_components_and_final(lowres_rgb, merge_radius_px=merge_radius_px)

    connectivity = 2 if use_8_connectivity else 1
    labeled = label(comp_mask, connectivity=connectivity)
    props = regionprops(labeled)

    # No tissue fallback
    if not props:
        scale_slide = min(1.0, 512 / max(H, W))
        rgb_512, mask_512 = letterbox_to_512(lowres_rgb, final_mask, scale=scale_slide)
        return [{
            "rgb_512": rgb_512,
            "mask_512": mask_512,
            "bbox": (0, H, 0, W),
            "area_px": 0,
            "chosen_level": chosen_level,
            "approx_mpp": approx_mpp,
        }]

    # Filter tiny components
    total_area = int(comp_mask.sum())
    min_area = max(1, int(total_area * min_area_ratio))
    kept = [p for p in props if p.area >= min_area]
    if not kept:
        kept = [max(props, key=lambda p: p.area)]
    kept.sort(key=lambda p: p.area, reverse=True)

    # Union bbox (over kept)
    rmin_all = min(p.bbox[0] for p in kept)
    cmin_all = min(p.bbox[1] for p in kept)
    rmax_all = max(p.bbox[2] for p in kept)
    cmax_all = max(p.bbox[3] for p in kept)

    union_h = rmax_all - rmin_all
    union_w = cmax_all - cmin_all
    union_area = max(1, union_h * union_w)
    sum_area = float(sum(p.area for p in kept))
    density = sum_area / union_area

    # Scale factor PER SLIDE based on padded union bbox (prevents per-component zoom)
    union_h_p = min(H, rmax_all + pad) - max(0, rmin_all - pad)
    union_w_p = min(W, cmax_all + pad) - max(0, cmin_all - pad)
    scale_slide = min(1.0, 512 / max(union_h_p, union_w_p))

    # Decide split vs single based on "big components" rule
    # area fraction relative to total kept tissue area
    area_fracs = [p.area / sum_area for p in kept]
    big_idxs = [i for i, f in enumerate(area_fracs) if f >= split_area_frac]

    # If we have at least TWO big components => we split into those big ones
    # (ignore smaller fragments in this mode)
    if len(big_idxs) >= 2:
        outputs = []
        for i in big_idxs:
            p = kept[i]
            r0, c0, r1, c1 = p.bbox
            rmin, rmax, cmin, cmax = _clamp_bbox(r0 - pad, r1 + pad, c0 - pad, c1 + pad, H, W)

            crop_rgb = lowres_rgb[rmin:rmax, cmin:cmax, :]
            crop_mask = final_mask[rmin:rmax, cmin:cmax]

            rgb_512, mask_512 = letterbox_to_512(crop_rgb, crop_mask, scale=scale_slide)
            outputs.append({
                "rgb_512": rgb_512,
                "mask_512": mask_512,
                "bbox": (rmin, rmax, cmin, cmax),
                "area_px": int(p.area),
                "chosen_level": chosen_level,
                "approx_mpp": approx_mpp,
            })
        return outputs

    # Otherwise we do NOT split.
    # Apply the density macro rule (optional – you already tuned this).
    # If dense, union crop is usually best; if sparse, policy decides.
    if small_component_policy == "largest" and density < density_single_thresh:
        # keep only largest component
        p = kept[0]
        r0, c0, r1, c1 = p.bbox
        rmin, rmax, cmin, cmax = _clamp_bbox(r0 - pad, r1 + pad, c0 - pad, c1 + pad, H, W)
        crop_rgb = lowres_rgb[rmin:rmax, cmin:cmax, :]
        crop_mask = final_mask[rmin:rmax, cmin:cmax]
        rgb_512, mask_512 = letterbox_to_512(crop_rgb, crop_mask, scale=scale_slide)
        return [{
            "rgb_512": rgb_512,
            "mask_512": mask_512,
            "bbox": (rmin, rmax, cmin, cmax),
            "area_px": int(p.area),
            "chosen_level": chosen_level,
            "approx_mpp": approx_mpp,
        }]

    # default: union crop (keeps the slide “whole”)
    rmin, rmax, cmin, cmax = _clamp_bbox(rmin_all - pad, rmax_all + pad, cmin_all - pad, cmax_all + pad, H, W)
    crop_rgb = lowres_rgb[rmin:rmax, cmin:cmax, :]
    crop_mask = final_mask[rmin:rmax, cmin:cmax]
    rgb_512, mask_512 = letterbox_to_512(crop_rgb, crop_mask, scale=scale_slide)
    return [{
        "rgb_512": rgb_512,
        "mask_512": mask_512,
        "bbox": (rmin, rmax, cmin, cmax),
        "area_px": int(sum_area),
        "chosen_level": chosen_level,
        "approx_mpp": approx_mpp,
    }]


# -------------------------
# CSV-driven runner
# -------------------------

def read_paths_from_csv(csv_path: str, col: str = "path"):
    paths = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if col not in reader.fieldnames:
            raise ValueError(f"CSV must contain column '{col}'. Found: {reader.fieldnames}")
        for row in reader:
            p = (row[col] or "").strip()
            if p:
                paths.append(p)
    return paths


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def process_from_csv(
    csv_path: str,
    output_root_img: str,
    output_root_mask: str,
    input_root_for_structure: str | None = None,
    **process_kwargs,
):
    """
    Processes only WSIs listed in csv_path, column 'path' (absolute paths).
    Keeps folder structure if you provide input_root_for_structure.

    Output paths:
      If input_root_for_structure is given:
        rel = relpath(wsi_path, input_root_for_structure)
        out_dir = join(output_root_img, dirname(rel))
      else:
        out_dir = output_root_img (flat)

    Filenames:
      1 output -> <slide>_512x512.png
      N outputs -> <slide>_partXX_512x512.png
    """
    ensure_dir(output_root_img)
    ensure_dir(output_root_mask)

    wsi_paths = read_paths_from_csv(csv_path, col="path")

    # If user did not provide a root, we can infer a common root (works only if paths share it)
    if input_root_for_structure is None and wsi_paths:
        input_root_for_structure = os.path.commonpath(wsi_paths)

    for wsi_path in wsi_paths:
        if not os.path.exists(wsi_path):
            print(f"Skipping missing: {wsi_path}")
            continue

        slide_name = os.path.splitext(os.path.basename(wsi_path))[0]

        if input_root_for_structure:
            rel = os.path.relpath(wsi_path, start=input_root_for_structure)
            rel_dir = os.path.dirname(rel)
            out_dir_img = os.path.join(output_root_img, rel_dir)
            out_dir_mask = os.path.join(output_root_mask, rel_dir)
        else:
            out_dir_img = output_root_img
            out_dir_mask = output_root_mask

        ensure_dir(out_dir_img)
        ensure_dir(out_dir_mask)

        print(f"Processing: {wsi_path}")
        try:
            parts = process_wsi_to_512_multi(wsi_path, **process_kwargs)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        if len(parts) == 1:
            rgb_path = os.path.join(out_dir_img, f"{slide_name}_512x512.png")
            mask_path = os.path.join(out_dir_mask, f"{slide_name}_512x512_mask.png")
            parts[0]["rgb_512"].save(rgb_path)
            parts[0]["mask_512"].save(mask_path)
            print(f"  Saved {rgb_path}")
            print(f"  Saved {mask_path}")
        else:
            for i, it in enumerate(parts, start=1):
                suffix = f"_part{i:02d}_512x512"
                rgb_path = os.path.join(out_dir_img, f"{slide_name}{suffix}.png")
                mask_path = os.path.join(out_dir_mask, f"{slide_name}{suffix}_mask.png")
                it["rgb_512"].save(rgb_path)
                it["mask_512"].save(mask_path)
                print(f"  Saved {rgb_path} (area={it['area_px']})")
                print(f"  Saved {mask_path}")

        print("  Done.")


# -------------------------
# Entry point
# -------------------------

if __name__ == "__main__":
    csv_path = "/storage/DSH/projects/iaso/diffusion_wholewsi/code/preprocessing/tcga_brca_histoqc_cleanList.csv"

    output_root_img = "/storage/DSH/projects/iaso/data/BRCA_prepr512_mergedRadiusPix20_density0.5_mpp4.0/img"
    output_root_mask = "/storage/DSH/projects/iaso/data/BRCA_prepr512_mergedRadiusPix20_density0.5_mpp4.0/masks"

    # This keeps relative folder structure under this root.
    # If your CSV paths are under /storage/DSH/projects/data/TCGA/BRCA/raw_data/wsi
    input_root_for_structure = "/storage/DSH/projects/data/TCGA/BRCA/raw_data/wsi"

    process_from_csv(
        csv_path=csv_path,
        output_root_img=output_root_img,
        output_root_mask=output_root_mask,
        input_root_for_structure=input_root_for_structure,

        # preprocessing params
        pad=20,
        target_mpp=4.0,
        merge_radius_px=20,
        min_area_ratio=0.01,
        use_8_connectivity=True,

        # merging/splitting controls
        density_single_thresh=0.75,

        # NEW splitting threshold:
        split_area_frac=0.25,              # split only if >=2 comps each >=25% of tissue
        small_component_policy="union",    # or "largest"
    )