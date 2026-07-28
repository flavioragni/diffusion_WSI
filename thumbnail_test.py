from __future__ import annotations

import os
import csv
import math
import argparse
import numpy as np
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

## Example call:
# python preprocess_wsi_components_to_tiles.py \
#   --csv-path /storage/DSH/projects/iaso/diffusion_wholewsi/code/preprocessing/tcga_brca_histoqc_cleanList.csv \
#   --output-root-img /storage/DSH/projects/iaso/data/BRCA_tiles/img \
#   --output-root-mask /storage/DSH/projects/iaso/data/BRCA_tiles/masks \
#   --input-root-for-structure /storage/DSH/projects/data/TCGA/BRCA/raw_data/wsi \
#   --target-size 256 \
#   --thumbnail-mpp 8.0 \
#   --tile-mpp 0.5 \
#   --min-tissue-frac 0.50 \
#   --pad 20 \
#   --min-area-ratio 0.01 \
#   --merge-radius-px 20 \
#   --use-8-connectivity \
#   --split-area-frac 0.25 \
#   --density-single-thresh 0.5 \
#   --small-component-policy union
##

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
    thumb_obj = reader.slide_thumbnail(resolution=float(thumbnail_mpp), units="mpp")
    return _thumb_to_rgb_uint8_np(thumb_obj)


# -------------------------
# Tissue detection utilities
# -------------------------

def make_masks_for_components_and_final(rgb_img_np: np.ndarray, merge_radius_px: int = 12):
    """
    - component_mask: Otsu + fill holes + optional closing
    - final_mask    : Otsu + dilation + fill holes
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


def make_tissue_mask_from_tile(tile_rgb: np.ndarray) -> np.ndarray:
    """
    Simple tile-level tissue mask using Otsu on grayscale.
    Returns uint8 mask in {0,255}.
    """
    gray = rgb2gray(tile_rgb).astype(np.float32)
    try:
        T = threshold_otsu(gray)
        base = gray < T
    except ValueError:
        # degenerate tile
        base = np.zeros(gray.shape, dtype=bool)

    base = binary_fill_holes(base)
    return (base.astype(np.uint8) * 255)


# -------------------------
# Geometry helpers
# -------------------------

def get_baseline_dimensions(reader: WSIReader) -> tuple[int, int]:
    """
    Returns (width, height) at baseline resolution.
    """
    info = getattr(reader, "info", None)

    # common TIAToolbox path
    if info is not None and hasattr(info, "slide_dimensions"):
        dims = info.slide_dimensions
        if len(dims) == 2:
            return int(dims[0]), int(dims[1])

    # fallback
    dims = reader.slide_dimensions()
    return int(dims[0]), int(dims[1])


def get_baseline_mpp(reader: WSIReader) -> tuple[float, float]:
    """
    Returns (mpp_x, mpp_y). Falls back to isotropic if only one value is available.
    """
    info = getattr(reader, "info", None)

    if info is not None and hasattr(info, "mpp") and info.mpp is not None:
        mpp = info.mpp
        if isinstance(mpp, (tuple, list)) and len(mpp) == 2:
            return float(mpp[0]), float(mpp[1])
        return float(mpp), float(mpp)

    raise ValueError("Could not determine baseline MPP from the WSI metadata.")


def thumb_bbox_to_baseline_bbox(
    bbox_thumb: tuple[int, int, int, int],
    thumb_shape_hw: tuple[int, int],
    base_dims_wh: tuple[int, int],
) -> tuple[int, int, int, int]:
    """
    Convert thumbnail bbox (rmin, rmax, cmin, cmax) to baseline bbox (x0, y0, x1, y1).
    """
    rmin, rmax, cmin, cmax = bbox_thumb
    Ht, Wt = thumb_shape_hw
    Wb, Hb = base_dims_wh

    x0 = int(round(cmin * Wb / Wt))
    x1 = int(round(cmax * Wb / Wt))
    y0 = int(round(rmin * Hb / Ht))
    y1 = int(round(rmax * Hb / Ht))

    x0 = max(0, min(Wb, x0))
    x1 = max(0, min(Wb, x1))
    y0 = max(0, min(Hb, y0))
    y1 = max(0, min(Hb, y1))

    return x0, y0, x1, y1


def add_pad_to_bbox_thumb(
    bbox_thumb: tuple[int, int, int, int],
    H: int,
    W: int,
    pad: int,
) -> tuple[int, int, int, int]:
    r0, r1, c0, c1 = bbox_thumb
    rmin = max(0, r0 - pad)
    rmax = min(H, r1 + pad)
    cmin = max(0, c0 - pad)
    cmax = min(W, c1 + pad)
    return (rmin, rmax, cmin, cmax)


# -------------------------
# Component selection logic
# -------------------------

def get_component_regions(
    wsi_path: str,
    thumbnail_mpp: float = 8.0,
    pad: int = 20,
    min_area_ratio: float = 0.01,
    merge_radius_px: int = 20,
    use_8_connectivity: bool = True,
    split_area_frac: float = 0.25,
    density_single_thresh: float = 0.5,
    small_component_policy: str = "union",
):
    """
    Finds components on thumbnail and returns a list of thumbnail-space bboxes
    following the same split / union logic as your original script.

    Returns list of dict with:
      - bbox_thumb: (rmin, rmax, cmin, cmax)
      - area_px
    """
    if small_component_policy not in ("union", "largest"):
        raise ValueError("small_component_policy must be 'union' or 'largest'")

    reader = WSIReader.open(wsi_path)
    thumb_rgb = get_thumbnail(reader, thumbnail_mpp=thumbnail_mpp)
    H, W = thumb_rgb.shape[:2]

    comp_mask, final_mask = make_masks_for_components_and_final(
        thumb_rgb, merge_radius_px=merge_radius_px
    )

    connectivity = 2 if use_8_connectivity else 1
    labeled = label(comp_mask, connectivity=connectivity)
    props = regionprops(labeled)

    if not props:
        return [{
            "bbox_thumb": (0, H, 0, W),
            "area_px": 0,
        }]

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

    if len(big_idxs) >= 2:
        outputs = []
        for i in big_idxs:
            p = kept[i]
            bbox_thumb = add_pad_to_bbox_thumb(
                (p.bbox[0], p.bbox[2], p.bbox[1], p.bbox[3]),
                H=H,
                W=W,
                pad=pad,
            )
            outputs.append({
                "bbox_thumb": bbox_thumb,
                "area_px": int(p.area),
            })
        return outputs

    if small_component_policy == "largest" and density < density_single_thresh:
        p = kept[0]
        bbox_thumb = add_pad_to_bbox_thumb(
            (p.bbox[0], p.bbox[2], p.bbox[1], p.bbox[3]),
            H=H,
            W=W,
            pad=pad,
        )
        return [{
            "bbox_thumb": bbox_thumb,
            "area_px": int(p.area),
        }]

    bbox_thumb = add_pad_to_bbox_thumb(
        (rmin_all, rmax_all, cmin_all, cmax_all),
        H=H,
        W=W,
        pad=pad,
    )
    return [{
        "bbox_thumb": bbox_thumb,
        "area_px": int(sum_area),
    }]


# -------------------------
# Tiling helpers
# -------------------------

def read_wsi_tile(
    reader: WSIReader,
    x_base: int,
    y_base: int,
    target_size: int,
    tile_mpp: float,
) -> np.ndarray:
    """
    Reads a tile from the WSI at requested MPP and output size target_size x target_size.
    Location is in baseline coordinates.
    """
    tile = reader.read_rect(
        location=(int(x_base), int(y_base)),
        size=(int(target_size), int(target_size)),
        resolution=float(tile_mpp),
        units="mpp",
        coord_space="baseline",
    )

    tile = _thumb_to_rgb_uint8_np(tile)
    return tile


def tile_component_from_wsi(
    reader: WSIReader,
    bbox_thumb: tuple[int, int, int, int],
    thumb_shape_hw: tuple[int, int],
    target_size: int,
    tile_mpp: float,
    min_tissue_frac: float,
):
    """
    Tile one component bbox directly from the WSI.

    Returns list of dicts with:
      - rgb: PIL.Image
      - mask: PIL.Image
      - tissue_frac: float
      - x_base, y_base
      - ix, iy
    """
    target_size = int(target_size)
    if not (0.0 <= min_tissue_frac <= 1.0):
        raise ValueError("min_tissue_frac must be in [0, 1].")

    base_dims = get_baseline_dimensions(reader)   # (Wb, Hb)
    mpp_x, mpp_y = get_baseline_mpp(reader)

    # convert component bbox from thumbnail to baseline coordinates
    x0, y0, x1, y1 = thumb_bbox_to_baseline_bbox(
        bbox_thumb=bbox_thumb,
        thumb_shape_hw=thumb_shape_hw,
        base_dims_wh=base_dims,
    )

    # physical tile coverage expressed in baseline pixels
    # output tile is target_size pixels at tile_mpp, so baseline extent is:
    tile_w_base = max(1, int(round(target_size * tile_mpp / mpp_x)))
    tile_h_base = max(1, int(round(target_size * tile_mpp / mpp_y)))

    comp_w_base = max(1, x1 - x0)
    comp_h_base = max(1, y1 - y0)

    n_tiles_x = max(1, math.ceil(comp_w_base / tile_w_base))
    n_tiles_y = max(1, math.ceil(comp_h_base / tile_h_base))

    kept_tiles = []

    for iy in range(n_tiles_y):
        for ix in range(n_tiles_x):
            tile_x = x0 + ix * tile_w_base
            tile_y = y0 + iy * tile_h_base

            # avoid requesting tiles entirely beyond the component
            if tile_x >= x1 or tile_y >= y1:
                continue

            tile_rgb = read_wsi_tile(
                reader=reader,
                x_base=tile_x,
                y_base=tile_y,
                target_size=target_size,
                tile_mpp=tile_mpp,
            )

            tissue_mask = make_tissue_mask_from_tile(tile_rgb)
            tissue_frac = float((tissue_mask > 0).mean())

            if tissue_frac < min_tissue_frac:
                continue

            kept_tiles.append({
                "rgb": Image.fromarray(tile_rgb),
                "mask": Image.fromarray(tissue_mask),
                "tissue_frac": tissue_frac,
                "x_base": tile_x,
                "y_base": tile_y,
                "ix": ix,
                "iy": iy,
            })

    return kept_tiles


# -------------------------
# CSV runner
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


def process_from_csv(
    csv_path: str,
    output_root_img: str,
    output_root_mask: str,
    input_root_for_structure: str | None = None,
    target_size: int = 256,
    thumbnail_mpp: float = 8.0,
    tile_mpp: float = 0.5,
    min_tissue_frac: float = 0.50,
    pad: int = 20,
    min_area_ratio: float = 0.01,
    merge_radius_px: int = 20,
    use_8_connectivity: bool = True,
    split_area_frac: float = 0.25,
    density_single_thresh: float = 0.5,
    small_component_policy: str = "union",
):
    ensure_dir(output_root_img)
    ensure_dir(output_root_mask)

    wsi_paths = read_paths_from_csv(csv_path, col="path")
    if input_root_for_structure is None and wsi_paths:
        input_root_for_structure = os.path.commonpath(wsi_paths)

    size_tag = f"{int(target_size)}x{int(target_size)}"
    tile_tag = f"tilempp{str(tile_mpp).replace('.', 'p')}"
    tissue_tag = f"tissue{int(round(min_tissue_frac * 100)):02d}"

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
            reader = WSIReader.open(wsi_path)
            thumb_rgb = get_thumbnail(reader, thumbnail_mpp=thumbnail_mpp)
            thumb_shape_hw = thumb_rgb.shape[:2]

            components = get_component_regions(
                wsi_path=wsi_path,
                thumbnail_mpp=thumbnail_mpp,
                pad=pad,
                min_area_ratio=min_area_ratio,
                merge_radius_px=merge_radius_px,
                use_8_connectivity=use_8_connectivity,
                split_area_frac=split_area_frac,
                density_single_thresh=density_single_thresh,
                small_component_policy=small_component_policy,
            )
        except Exception as e:
            print(f"  ERROR during component extraction: {e}")
            continue

        total_saved = 0

        for comp_idx, comp in enumerate(components, start=1):
            try:
                tiles = tile_component_from_wsi(
                    reader=reader,
                    bbox_thumb=comp["bbox_thumb"],
                    thumb_shape_hw=thumb_shape_hw,
                    target_size=target_size,
                    tile_mpp=tile_mpp,
                    min_tissue_frac=min_tissue_frac,
                )
            except Exception as e:
                print(f"  ERROR during tiling component {comp_idx}: {e}")
                continue

            if not tiles:
                print(f"  Component {comp_idx}: no tiles kept after tissue threshold.")
                continue

            for t in tiles:
                suffix = (
                    f"_part{comp_idx:02d}"
                    f"_x{t['ix']:03d}_y{t['iy']:03d}"
                    f"_{size_tag}_{tile_tag}_{tissue_tag}"
                )

                rgb_path = os.path.join(out_dir_img, f"{slide_name}{suffix}.png")
                mask_path = os.path.join(out_dir_mask, f"{slide_name}{suffix}_mask.png")

                t["rgb"].save(rgb_path)
                t["mask"].save(mask_path)

                total_saved += 1
                print(f"  Saved {rgb_path} (tissue_frac={t['tissue_frac']:.3f})")
                print(f"  Saved {mask_path}")

        print(f"  Done. Total kept tiles: {total_saved}")


# -------------------------
# CLI
# -------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Split WSI into components on thumbnail, then tile each component from WSI."
    )

    parser.add_argument("--csv-path", type=str, required=True, help="CSV containing a 'path' column.")
    parser.add_argument("--output-root-img", type=str, required=True, help="Output root for image tiles.")
    parser.add_argument("--output-root-mask", type=str, required=True, help="Output root for mask tiles.")
    parser.add_argument("--input-root-for-structure", type=str, default=None, help="Optional root to preserve folder structure.")

    # target tile size
    parser.add_argument("--target-size", type=int, default=256, help="Tile output size in pixels.")

    # thumbnail / component parameters
    parser.add_argument("--thumbnail-mpp", type=float, default=8.0, help="MPP for thumbnail used for component detection.")
    parser.add_argument("--pad", type=int, default=20, help="Padding around selected component bbox in thumbnail pixels.")
    parser.add_argument("--min-area-ratio", type=float, default=0.01, help="Minimum component area ratio on thumbnail.")
    parser.add_argument("--merge-radius-px", type=int, default=20, help="Closing radius for merging nearby tissue fragments on thumbnail.")
    parser.add_argument("--use-8-connectivity", action="store_true", help="Use 8-connectivity for connected components.")
    parser.add_argument("--split-area-frac", type=float, default=0.25, help="Split only if >=2 components each >= this area fraction.")
    parser.add_argument("--density-single-thresh", type=float, default=0.5, help="Used only when not splitting.")
    parser.add_argument(
        "--small-component-policy",
        type=str,
        choices=["union", "largest"],
        default="union",
        help="When not splitting, keep union or largest component.",
    )

    # new parameters requested
    parser.add_argument("--tile-mpp", type=float, default=0.5, help="MPP for actual WSI tiling.")
    parser.add_argument("--min-tissue-frac", type=float, default=0.50, help="Minimum fraction of tissue required to keep a tile.")

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    process_from_csv(
        csv_path=args.csv_path,
        output_root_img=args.output_root_img,
        output_root_mask=args.output_root_mask,
        input_root_for_structure=args.input_root_for_structure,
        target_size=args.target_size,
        thumbnail_mpp=args.thumbnail_mpp,
        tile_mpp=args.tile_mpp,
        min_tissue_frac=args.min_tissue_frac,
        pad=args.pad,
        min_area_ratio=args.min_area_ratio,
        merge_radius_px=args.merge_radius_px,
        use_8_connectivity=args.use_8_connectivity,
        split_area_frac=args.split_area_frac,
        density_single_thresh=args.density_single_thresh,
        small_component_policy=args.small_component_policy,
    )