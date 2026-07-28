import os
import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import disk
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes, binary_dilation as ndi_binary_dilation

from tiatoolbox.wsicore.wsireader import WSIReader


# -------------------------
# Tissue detection utilities
# -------------------------

def make_masks_for_components_and_final(rgb_img_np: np.ndarray):
    """
    Build two masks from a low-mag RGB image:
      - component_mask: Otsu + fill holes (NO dilation) -> good to separate tissues
      - final_mask    : Otsu + dilation (r=5, iter=1) + fill holes -> cleaner edges for saving

    Returns:
      component_mask (bool HxW), final_mask (bool HxW)
    """
    gray = rgb2gray(rgb_img_np).astype(np.float32)
    T = threshold_otsu(gray)
    base = gray < T  # True = tissue

    component_mask = binary_fill_holes(base)

    selem = disk(5).astype(bool)
    dil = ndi_binary_dilation(base, structure=selem, iterations=1)
    final_mask = binary_fill_holes(dil)

    return component_mask, final_mask


def bbox_from_mask(mask: np.ndarray, pad: int = 0):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        h, w = mask.shape
        return 0, h, 0, w
    rmin = max(0, ys.min() - pad)
    rmax = min(mask.shape[0], ys.max() + 1 + pad)
    cmin = max(0, xs.min() - pad)
    cmax = min(mask.shape[1], xs.max() + 1 + pad)
    return rmin, rmax, cmin, cmax


def letterbox_to_512(rgb_np: np.ndarray, mask_np: np.ndarray):
    """
    Aspect-ratio preserving fit into 512x512 (black bg for RGB and mask).
    """
    h, w = rgb_np.shape[:2]
    target = 512

    scale = target / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    rgb_resized = Image.fromarray(rgb_np).resize((new_w, new_h), resample=Image.BILINEAR)
    mask_resized = Image.fromarray((mask_np.astype(np.uint8) * 255)).resize((new_w, new_h), resample=Image.NEAREST)

    canvas_rgb = np.zeros((target, target, 3), dtype=np.uint8) * 255
    canvas_mask = np.zeros((target, target), dtype=np.uint8)

    y0 = (target - new_h) // 2
    x0 = (target - new_w) // 2
    canvas_rgb[y0:y0+new_h, x0:x0+new_w, :] = np.asarray(rgb_resized)
    canvas_mask[y0:y0+new_h, x0:x0+new_w] = np.asarray(mask_resized)

    return Image.fromarray(canvas_rgb), Image.fromarray(canvas_mask)


# -------------------------
# WSI processing (single slide → possibly multiple crops)
# -------------------------

def process_wsi_to_512_multi(
    wsi_path: str,
    pad: int = 20,
    min_area_ratio: float = 0.01,  # ignore components smaller than 1% of total mask area
):
    """
    Load a WSI, read full coarsest level, detect tissues, split by connected components,
    crop each component with padding, letterbox to 512, and return a list of crops.

    Returns:
      items: list of dicts, each with:
        - 'rgb_512'  : PIL.Image
        - 'mask_512' : PIL.Image
        - 'bbox'     : (rmin,rmax,cmin,cmax) in coarsest-level coordinates
        - 'area_px'  : component area in pixels (component mask)
    """
    # open slide
    wsi = WSIReader.open(wsi_path)

    # pyramid info
    level_dims = wsi.info.level_dimensions          # ((W0,H0), (W1,H1), ...)
    num_levels = len(level_dims)
    coarsest_level = num_levels - 1
    coarse_w, coarse_h = level_dims[coarsest_level]

    # read entire coarsest level
    lowres_rgba = wsi.read_region(location=(0, 0), level=coarsest_level, size=(coarse_w, coarse_h))
    lowres_rgb = np.asarray(lowres_rgba)[..., :3]

    # orientation sanity (rare, defensive)
    H_read, W_read = lowres_rgb.shape[:2]
    if (H_read == coarse_w) and (W_read == coarse_h) and (H_read != coarse_h):
        lowres_rgb = np.rot90(lowres_rgb, k=3)

    # build masks
    component_mask, final_mask = make_masks_for_components_and_final(lowres_rgb)

    # connected components on the *component mask* (to avoid fusing nearby tissues)
    labeled = label(component_mask, connectivity=1)  # 4-connected
    props = regionprops(labeled)

    if len(props) == 0:
        # no tissue -> just fall back to whole slide
        rmin, rmax, cmin, cmax = bbox_from_mask(final_mask, pad=pad)
        crop_rgb = lowres_rgb[rmin:rmax, cmin:cmax, :]
        crop_mask = final_mask[rmin:rmax, cmin:cmax]
        rgb_512, mask_512 = letterbox_to_512(crop_rgb, crop_mask)
        return [{"rgb_512": rgb_512, "mask_512": mask_512, "bbox": (rmin, rmax, cmin, cmax), "area_px": 0}]

    # filter tiny components
    total_mask_area = int(component_mask.sum())
    min_area = int(total_mask_area * min_area_ratio)
    kept = [p for p in props if p.area >= max(min_area, 1)]

    # if after filtering we're down to none, keep the largest
    if len(kept) == 0:
        kept = [max(props, key=lambda p: p.area)]

    # sort by descending area for nicer naming (part01 = biggest)
    kept.sort(key=lambda p: p.area, reverse=True)

    items = []
    for p in kept:
        rmin, cmin, rmax, cmax = p.bbox  # skimage gives (min_row, min_col, max_row, max_col)
        # add padding and clamp to bounds
        rmin, rmax, cmin, cmax = bbox_from_mask(labeled == p.label, pad=pad)

        # crop from RGB and from the *final* mask (for nicer borders)
        crop_rgb = lowres_rgb[rmin:rmax, cmin:cmax, :]
        crop_mask = final_mask[rmin:rmax, cmin:cmax]

        rgb_512, mask_512 = letterbox_to_512(crop_rgb, crop_mask)
        items.append({
            "rgb_512": rgb_512,
            "mask_512": mask_512,
            "bbox": (rmin, rmax, cmin, cmax),
            "area_px": int(p.area),
        })

    return items


# -------------------------
# Batch runner (recursive)
# -------------------------

def is_wsi_file(filename: str):
    exts = [".svs", ".tif", ".tiff", ".ndpi", ".svslide", ".mrxs", ".scn", ".bif", ".vms", ".vmu"]
    lower = filename.lower()
    return any(lower.endswith(e) for e in exts)


def process_folder(
    input_root: str,
    output_root_img: str,
    output_root_mask: str,
    pad: int = 20,
    min_area_ratio: float = 0.01,
):
    os.makedirs(output_root_img, exist_ok=True)
    os.makedirs(output_root_mask, exist_ok=True)

    for dirpath, _, filenames in os.walk(input_root):
        for fname in filenames:
            if not is_wsi_file(fname):
                continue

            wsi_path = os.path.join(dirpath, fname)
            slide_name = os.path.splitext(fname)[0]

            print(f"Processing: {wsi_path}")
            try:
                parts = process_wsi_to_512_multi(
                    wsi_path=wsi_path,
                    pad=pad,
                    min_area_ratio=min_area_ratio,
                )
            except Exception as e:
                print(f"  ERROR processing {wsi_path}: {e}")
                continue

            if len(parts) == 1:
                # single tissue
                rgb_path = os.path.join(output_root_img, f"{slide_name}_512x512.png")
                mask_path = os.path.join(output_root_mask, f"{slide_name}_512x512_mask.png")
                parts[0]["rgb_512"].save(rgb_path)
                parts[0]["mask_512"].save(mask_path)
                print(f"  Saved {rgb_path}")
                print(f"  Saved {mask_path}")
            else:
                # multiple tissues -> save *_partXX_*
                for i, it in enumerate(parts, start=1):
                    suffix = f"_part{i:02d}_512x512"
                    rgb_path = os.path.join(output_root_img, f"{slide_name}{suffix}.png")
                    mask_path = os.path.join(output_root_mask, f"{slide_name}{suffix}_mask.png")
                    it["rgb_512"].save(rgb_path)
                    it["mask_512"].save(mask_path)
                    print(f"  Saved {rgb_path}  (area={it['area_px']})")
                    print(f"  Saved {mask_path}")

            print("  Done.")

# -------------------------
# Example CLI entry point
# -------------------------

if __name__ == "__main__":
    input_root       = "/storage/DSH/projects/iaso/data/test-wsi-brca/8709d0f0-7e55-44e6-b1fe-cd7c8ae6bf67"         # can contain subfolders
    output_root_img  = "/storage/DSH/projects/iaso/data/thumbnail_test"
    output_root_mask = "/storage/DSH/projects/iaso/data/thumbnail_test"

    process_folder(
        input_root=input_root,
        output_root_img=output_root_img,
        output_root_mask=output_root_mask,
        pad=20,
        min_area_ratio=0.01,  # set higher (e.g., 0.03) to drop very small fragments
    )
