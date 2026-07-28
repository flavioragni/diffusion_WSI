import lightning as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger

import os
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import wandb

from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from dataset import DiffusionSchedule


def make_timestep_schedule(num_timesteps: int, sample_steps: int) -> list[int]:
    if sample_steps >= num_timesteps:
        return list(range(num_timesteps - 1, -1, -1))

    ts = np.linspace(0, num_timesteps - 1, sample_steps, dtype=np.int64)
    ts = np.unique(ts)
    ts = ts[::-1].tolist()
    if ts[-1] != 0:
        ts.append(0)
    return ts


@torch.no_grad()
def predict_eps(lit_model, x_t, t, class_id, guidance_scale: float) -> torch.Tensor:
    denoiser = lit_model.model
    if (class_id is None) or (guidance_scale == 1.0):
        return denoiser(x_t, time=t, classes=class_id, cond_drop_prob=0.0)

    eps_uncond = denoiser(x_t, time=t, classes=None, cond_drop_prob=0.0)
    eps_cond = denoiser(x_t, time=t, classes=class_id, cond_drop_prob=0.0)
    return eps_uncond + guidance_scale * (eps_cond - eps_uncond)


@torch.no_grad()
def ddim_sample(
    lit_model,
    schedule: DiffusionSchedule,
    batch_size: int,
    image_size: int,
    channels: int,
    class_id: int | None,
    guidance_scale: float,
    sample_steps: int,
    eta: float,
    clip_denoised: bool,
    device: torch.device,
) -> torch.Tensor:
    num_timesteps = schedule.num_timesteps
    abar = schedule.alphas_cumprod.to(device)

    timesteps = make_timestep_schedule(num_timesteps, sample_steps)

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

        sigma = eta * torch.sqrt((1.0 - abar_prev) / (1.0 - abar_t)) * torch.sqrt(
            1.0 - (abar_t / abar_prev)
        )

        dir_coeff = torch.sqrt(torch.clamp(1.0 - abar_prev - sigma**2, min=0.0))
        x_prev = torch.sqrt(abar_prev) * x0 + dir_coeff * eps

        if t_prev >= 0 and eta > 0.0:
            x_prev = x_prev + sigma * torch.randn_like(x)

        x = x_prev

    return x


@torch.no_grad()
def ddpm_sample_full(
    lit_model,
    schedule: DiffusionSchedule,
    batch_size: int,
    image_size: int,
    channels: int,
    class_id: int | None,
    guidance_scale: float,
    device: torch.device,
) -> torch.Tensor:
    num_timesteps = schedule.num_timesteps
    betas = schedule.betas.to(device)
    alphas = schedule.alphas.to(device)
    abar = schedule.alphas_cumprod.to(device)

    x = torch.randn(batch_size, channels, image_size, image_size, device=device)
    y = None if class_id is None else torch.full((batch_size,), int(class_id), device=device, dtype=torch.long)

    for t_idx in range(num_timesteps - 1, -1, -1):
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


class DiffusionSampleGridCallback(Callback):
    """
    Logs a grid of generated samples every N epochs (on_train_epoch_end).
    Epoch counting uses 1-based human logic: logs after epochs N, 2N, 3N, ...
    (i.e. epoch indices N-1, 2N-1, 3N-1, ...).
    """

    def __init__(
        self,
        num_timesteps: int,
        schedule: str,
        beta_start: float,
        beta_end: float,
        cosine_s: float,
        image_size: int,
        channels: int,
        every_n_epochs: int = 1,  # <-- changed from every_n_steps
        num_images: int = 5,
        sample_steps: int = 50,
        eta: float = 0.0,
        clip_denoised: bool = True,
        guidance_scale: float = 1.0,
        class_id: int | None = None,
        sampler: str = "ddpm",
        log_key: str = "samples/grid",
    ):
        super().__init__()
        self.schedule = DiffusionSchedule(
            num_timesteps=num_timesteps,
            schedule=schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            cosine_s=cosine_s,
        )
        self.image_size = int(image_size)
        self.channels = int(channels)

        self.every_n_epochs = int(every_n_epochs)
        if self.every_n_epochs < 1:
            raise ValueError("every_n_epochs must be >= 1")

        self.num_images = int(num_images)
        self.sample_steps = int(sample_steps)
        self.eta = float(eta)
        self.clip_denoised = bool(clip_denoised)
        self.guidance_scale = float(guidance_scale)
        self.class_id = class_id
        self.sampler = sampler
        self.log_key = log_key

        self._last_logged_epoch = -1

    @torch.no_grad()
    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if not trainer.is_global_zero:
            return
        if getattr(trainer, "sanity_checking", False):
            return

        epoch = int(trainer.current_epoch)  # 0-indexed
        # Log AFTER epochs: N, 2N, 3N... (human counting)
        if ((epoch + 1) % self.every_n_epochs) != 0:
            return
        if epoch == self._last_logged_epoch:
            return

        self._last_logged_epoch = epoch
        step = int(trainer.global_step)  # keep W&B step aligned with training progress

        device = pl_module.device
        if self.sampler == "ddpm":
            samples = ddpm_sample_full(
                lit_model=pl_module,
                schedule=self.schedule,
                batch_size=self.num_images,
                image_size=self.image_size,
                channels=self.channels,
                class_id=self.class_id,
                guidance_scale=self.guidance_scale,
                device=device,
            )
        else:
            samples = ddim_sample(
                lit_model=pl_module,
                schedule=self.schedule,
                batch_size=self.num_images,
                image_size=self.image_size,
                channels=self.channels,
                class_id=self.class_id,
                guidance_scale=self.guidance_scale,
                sample_steps=self.sample_steps,
                eta=self.eta,
                clip_denoised=self.clip_denoised,
                device=device,
            )

        samples_01 = (samples.clamp(-1, 1) + 1) * 0.5
        grid = make_grid(samples_01, nrow=self.num_images)

        # Prefer WandbLogger if present
        for logger in (getattr(trainer, "loggers", None) or []):
            if isinstance(logger, WandbLogger):
                logger.experiment.log({self.log_key: wandb.Image(grid)}, step=step)
                return

        # Fallback if wandb is initialized without Lightning logger
        if wandb.run is not None:
            wandb.log({self.log_key: wandb.Image(grid)}, step=step)


# You can create your own callback
class MyPrintingCallback(Callback):
    def __init__(self, lr, bs, do, wd, config):
        self.lr = lr
        self.bs = bs
        self.do = do
        self.wd = wd
        self.config = config

    # Define a function to do something when training starts
    def on_train_start(self, trainer, pl_module):
        # Print training parameters
        print(
            f"""\nParameters:\n
              Num channels: {self.config.training['NUM_CHANNELS']},\n
              Learning rate: {self.lr},\n
              Dropout: {self.do},\n
              Batch size: {self.bs},\n
              Weight decay: {self.wd},\n
              Num epochs: {self.config.training['NUM_EPOCHS']},\n
              Patience: {self.config.training['PATIENCE']},\n
              Pretrained: {self.config.model['PRETRAINED']},\n
              Fine-tuning: {self.config.model['FINE_TUNE']},\n
              FastDevRun: {self.config.training['FAST_DEV_RUN']}"""
        )
        print("Starting to train!")