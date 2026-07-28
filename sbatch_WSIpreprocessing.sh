# dataset.py
import os
import glob
from typing import Optional, Tuple, Dict, List
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import lightning as L
import torchvision.transforms as T


# -----------------------------
# Diffusion schedule utilities
# -----------------------------

def _beta_schedule_linear(T: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    """Classic linear beta schedule (Ho et al., 2020)."""
    return torch.linspace(beta_start, beta_end, T, dtype=torch.float32)


def _beta_schedule_cosine(T: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule from Nichol & Dhariwal (2021).
    Returns betas in (0,1). More signal preserved at early timesteps.
    """
    steps = T + 1
    t = torch.linspace(0, T, steps, dtype=torch.float32)
    f = torch.cos(((t / T) + s) / (1 + s) * (np.pi / 2)) ** 2
    alpha_bar = f / f[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clamp(betas, 1e-8, 0.999)


class DiffusionSchedule:
    """
    Holds betas, alphas, and cumulative products for q(x_t | x0).
    Use .q_sample(x0, t, noise) to generate x_t = √ᾱ_t * x0 + √(1-ᾱ_t) * ε
    """
    def __init__(
        self,
        num_timesteps: int = 1000,
        schedule: str = "linear",      # "linear" or "cosine"
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        cosine_s: float = 0.008,
    ):
        if schedule == "linear":
            betas = _beta_schedule_linear(num_timesteps, beta_start, beta_end)
        elif schedule == "cosine":
            betas = _beta_schedule_cosine(num_timesteps, cosine_s)
        else:
            raise ValueError(f"Unknown schedule '{schedule}'")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.num_timesteps = int(num_timesteps)
        self.betas = betas          # [T]
        self.alphas = alphas        # [T]
        self.alphas_cumprod = alphas_cumprod  # [T]

    def sample_t(self, batch_size: int) -> torch.Tensor:
        """Uniformly sample timesteps in [0, T-1]. Returns LongTensor [B]."""
        return torch.randint(0, self.num_timesteps, (batch_size,), dtype=torch.long)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        x0:    [B,C,H,W] in [-1,1] or [0,1]
        t:     [B] (long)
        noise: [B,C,H,W] ~ N(0,1)
        returns x_t with shape [B,C,H,W]
        """
        device = x0.device
        ab = self.alphas_cumprod.to(device)[t]               # [B]
        sqrt_ab = torch.sqrt(ab).view(-1, 1, 1, 1)           # [B,1,1,1]
        sqrt_one_minus_ab = torch.sqrt(1.0 - ab).view(-1, 1, 1, 1)
        return sqrt_ab * x0 + sqrt_one_minus_ab * noise


# -----------------------------
# File discovery
# -----------------------------

def _list_images(root_dir: str, exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg")) -> List[str]:
    files: List[str] = []
    for e in exts:
        files.extend(glob.glob(os.path.join(root_dir, f"**/*{e}"), recursive=True))
    files = sorted(files)
    if len(files) == 0:
        raise RuntimeError(f"No images found in '{root_dir}' with extensions {exts}")
    return files


# -----------------------------
# Dataset
# -----------------------------

class WSIDiffusionDataset(Dataset):
    """
    Dataset for WSI tiles (e.g., 512x512 PNGs) to train diffusion UNet (noise prediction).
    Returns a dict per item; use the custom collate to convert to the tuple the model expects.

    Output per sample (dict keys):
        - x_t:      [C,H,W]   noisy image at timestep t (float32, ~[-1,1])
        - t:        [1]       timestep (LongTensor)
        - noise:    [C,H,W]   epsilon ~ N(0,1) used to create x_t
        - class_id: Optional[LongTensor scalar] (or None) for conditional training
        - x0:       [C,H,W]   clean normalized image (for optional logging)
        - path:     str       original file path (debug)

    Use diffusion_collate() to yield (x_t, t, noise, class_id) batches for the model.
    """

    def __init__(
        self,
        root_dir: str,
        schedule: DiffusionSchedule,
        split: str = "train",
        class_map: Optional[Dict[str, int]] = None,  # {basename_without_ext: class_id}
        image_size: int = 512,
        augment: bool = True,
        files: Optional[List[str]] = None,           # if you want to pass a pre-split file list
    ):
        super().__init__()
        self.root_dir = root_dir
        self.schedule = schedule
        self.split = split
        self.class_map = class_map
        self.image_size = int(image_size)

        self.files = files if files is not None else _list_images(root_dir)

        # transforms: ToTensor -> [-1,1] remap; optional augmentations on train
        base = [T.ToTensor(), T.Resize((self.image_size, self.image_size), antialias=True)]
        aug = []
        if split == "train" and augment:
            aug = [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(degrees=(90,90)), #T.RandomRotation(degrees=90),
                T.RandomRotation(degrees=(-90, -90)),
                #T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            ]
        self.transform = T.Compose(aug + base)

    def __len__(self) -> int:
        return len(self.files)

    def _load_x0(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        x0 = self.transform(img)           # [C,H,W] in [0,1]
        x0 = x0 * 2.0 - 1.0                # map to [-1,1]
        return x0

    def _class_id_from_path(self, path: str) -> Optional[torch.Tensor]:
        if self.class_map is None:
            return None
        base = os.path.basename(path)
        key = os.path.splitext(base)[0]
        cid = self.class_map.get(key, 0)
        return torch.tensor(cid, dtype=torch.long)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        x0 = self._load_x0(path)                  # [C,H,W], [-1,1]
        noise = torch.randn_like(x0)              # [C,H,W]
        # Sample a random timestep t
        t = self.schedule.sample_t(1)             # [1] Long
        # Generate the noisy image based on t noise
        x_t = self.schedule.q_sample(
            x0=x0.unsqueeze(0),                   # [1,C,H,W]
            t=t,                                  # [1]
            noise=noise.unsqueeze(0),             # [1,C,H,W]
        ).squeeze(0)                               # [C,H,W]
        # Add class for conditioned training if needed
        class_id = self._class_id_from_path(path)
        return {
            "x_t": x_t,
            "t": t,                      # [1] long (will be concatenated to [B])
            "noise": noise,
            "class_id": class_id,        # Long scalar or None
            "x0": x0,                    # handy for logging
            "path": path
        }


# -----------------------------
# Collate function
# -----------------------------

def diffusion_collate(batch: List[Dict]):
    """
    Convert list of dicts from WSIDiffusionDataset into the exact tuple
    expected by the Lightning model: (x_t, t, noise, class_id)

    Returns:
        x_t:     [B,C,H,W]
        t:       [B] (long)
        noise:   [B,C,H,W]
        class_id:[B] (long) or None
    """
    x_t = torch.stack([b["x_t"] for b in batch], dim=0)         # [B,C,H,W]
    noise = torch.stack([b["noise"] for b in batch], dim=0)     # [B,C,H,W]
    t = torch.cat([b["t"] for b in batch], dim=0).long()        # [B]

    cls_list = [b["class_id"] for b in batch]
    if any(c is None for c in cls_list):
        class_id = None
    else:
        class_id = torch.stack(cls_list, dim=0).long()          # [B]

    return x_t, t, noise, class_id

# -----------------------------
# DataModule (train only)
# -----------------------------

class WSIDiffusionDataModule(L.LightningDataModule):
    """
    Train-only datamodule for diffusion training.
    Validation and test dataloaders are intentionally disabled.
    """

    def __init__(
        self,
        data_root: str,
        batch_size: int = 16,
        num_workers: int = 8,

        test_frac: float = 0.2,            # kept for backward compatibility, unused in train-only mode
        test_split_csv_path: Optional[str] = None,  # kept for backward compatibility, unused

        image_size: int = 512,

        # diffusion schedule params
        num_timesteps: int = 1000,
        schedule: str = "linear",        # "linear" or "cosine"
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        cosine_s: float = 0.008,

        # conditioning
        class_map: Optional[Dict[str, int]] = None,
        augment_train: bool = True,
        seed: int = 42,

        overwrite_split_csv: bool = True,  # kept for backward compatibility, unused
    ):
        super().__init__()
        self.data_root = data_root
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)

        self.test_frac = float(test_frac)
        if not (0.0 <= self.test_frac < 1.0):
            raise ValueError("test_frac must be in [0, 1).")

        self.test_split_csv_path = test_split_csv_path
        self.overwrite_split_csv = bool(overwrite_split_csv)

        self.image_size = int(image_size)
        self.class_map = class_map
        self.augment_train = bool(augment_train)
        self.seed = int(seed)

        # shared diffusion schedule across splits
        self.schedule = DiffusionSchedule(
            num_timesteps=num_timesteps,
            schedule=schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            cosine_s=cosine_s,
        )

        # filled in setup()
        self.train_ds: Optional[WSIDiffusionDataset] = None

        # keep split info for reproducibility / debugging
        self._all_files: Optional[List[str]] = None
        self._train_idx: Optional[np.ndarray] = None
        self._did_setup: bool = False

    def setup(self, stage: Optional[str] = None):
        # Avoid re-splitting if setup is called multiple times
        if self._did_setup:
            return

        all_files = _list_images(self.data_root)  # sorted
        train_idx = np.arange(len(all_files))
        train_files = all_files

        # build datasets
        self.train_ds = WSIDiffusionDataset(
            root_dir=self.data_root,
            schedule=self.schedule,
            split="train",
            class_map=self.class_map,
            image_size=self.image_size,
            augment=self.augment_train,
            files=train_files,
        )

        # store split info
        self._all_files = all_files
        self._train_idx = train_idx
        self._did_setup = True

    def train_dataloader(self):
        if self.train_ds is None:
            raise RuntimeError("train_ds is not initialized. Did you call setup()?")
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(self.num_workers > 0),
            collate_fn=diffusion_collate,
        )

    def val_dataloader(self):
        # Lightning still validates dataloader shape/type even when validation is disabled.
        # Return an empty loader instead of None/[] to keep train-only runs valid.
        if self.train_ds is None:
            raise RuntimeError("train_ds is not initialized. Did you call setup()?")
        empty_ds = Subset(self.train_ds, [])
        return DataLoader(
            empty_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            collate_fn=diffusion_collate,
        )

    def test_dataloader(self):
        if self.train_ds is None:
            raise RuntimeError("train_ds is not initialized. Did you call setup()?")
        empty_ds = Subset(self.train_ds, [])
        return DataLoader(
            empty_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            collate_fn=diffusion_collate,
        )
