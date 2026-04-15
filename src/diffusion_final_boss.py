"""
diffusion_artbench.py
=====================
DDPM with optional Classifier-Free Guidance (CFG) for ArtBench-10 (32x32).

Key improvements over the starter notebook
-------------------------------------------
1. Class-conditional U-Net with Classifier-Free Guidance (CFG).
   At training time the class label is dropped with probability p_uncond,
   teaching the model both conditional and unconditional denoising.
   At sampling time the two predictions are interpolated:
       eps_cfg = eps_uncond + w * (eps_cond - eps_uncond)
   This is the dominant approach in modern diffusion (Ho & Salimans, 2022).

2. Exponential Moving Average (EMA) of model weights.
   EMA weights produce significantly smoother samples and are standard in
   every serious DDPM implementation.

3. DDIM deterministic sampler (Song et al., 2020).
   Allows high-quality sampling in 50-100 steps instead of 1000, making
   the mandatory 5000-sample evaluation loop tractable.

4. Ablation-ready config dataclass.
   Each experiment is fully described by a Config object, so ablations can
   be run by changing one field.

Usage
-----
# Quick smoke test on subset:
    python diffusion_artbench.py --mode train --config small --epochs 50

# Full training with best config:
    python diffusion_artbench.py --mode train --config best --full_dataset --epochs 500

# Evaluate a saved checkpoint:
    python diffusion_artbench.py --mode eval --checkpoint models/best_ema.pt

References
----------
Ho et al. (2020) DDPM - NeurIPS
Ho & Salimans (2022) Classifier-Free Diffusion Guidance
Song et al. (2020) DDIM - ICLR 2021
Nichol & Dhariwal (2021) Improved DDPM - cosine schedule
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm


@dataclass
class Config:
    # Dataset
    image_size: int = 32
    num_classes: int = 10
    batch_size: int = 128
    num_workers: int = 4

    # Diffusion schedule
    T: int = 1000
    cosine_s: float = 0.008            # offset for cosine schedule

    # U-Net architecture
    model_channels: int = 128          # base channel width C
    channel_mult: tuple = (1, 2, 4)   # multipliers per resolution level
    num_res_blocks: int = 2
    use_attention: bool = True         # self-attention at 8x8 bottleneck

    # Classifier-Free Guidance
    use_cfg: bool = True
    p_uncond: float = 0.10             # label drop probability during training
    cfg_scale: float = 3.0            # w at sampling time; 1.0 = no guidance

    # Training
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    epochs: int = 100
    warmup_epochs: int = 5

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.9999

    # Sampling
    ddim_steps: int = 100             # steps for DDIM sampler
    ddim_eta: float = 0.0             # 0 = deterministic DDIM

    # Paths
    project_root: str = "../DATA"
    subset_csv: str = "training_20_percent.csv"
    model_dir: str = "models"
    history_dir: str = "histories"
    sample_dir: str = "samples"

    # Run name (auto-derived if empty)
    run_name: str = ""

    def derive_run_name(self) -> str:
        cfg_tag = "cfg" if self.use_cfg else "uncond"
        ema_tag = "ema" if self.use_ema else "noema"
        return (
            f"ddpm_C{self.model_channels}"
            f"_{cfg_tag}_w{self.cfg_scale}"
            f"_{ema_tag}"
            f"_ep{self.epochs}"
        )

# Predefined configs for ablation study
CONFIGS: dict[str, Config] = {
    # Baseline: unconditional, no EMA (replicates starter notebook)
    "baseline": Config(
        model_channels=128, use_cfg=False, use_ema=False,
        epochs=100, run_name="baseline"
    ),
    # Small: fast iteration on subset, conditional + EMA
    "small": Config(
        model_channels=64, use_cfg=True, cfg_scale=3.0,
        use_ema=True, epochs=50, run_name="small"
    ),
    # Medium: recommended for subset ablation
    "medium": Config(
        model_channels=128, use_cfg=True, cfg_scale=3.0,
        use_ema=True, epochs=100, run_name="medium"
    ),
    # Best: full training run
    "best": Config(
        model_channels=128, use_cfg=True, cfg_scale=4.0,
        use_ema=True, epochs=500, batch_size=128, run_name="best"
    ),
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def load_artbench_splits(kaggle_root: Path):
    """Load the HuggingFace-style ArtBench splits via the provided helper."""
    scripts_dir = kaggle_root.parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from artbench_local_dataset import load_kaggle_artbench10_splits  # noqa
    return load_kaggle_artbench10_splits(kaggle_root)


def load_subset_ids(csv_path: Path, col: str = "train_id_original") -> list[int]:
    ids: list[int] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            v = row.get(col, "").strip()
            if v:
                ids.append(int(v))
    if not ids:
        raise ValueError(f"No IDs found in {csv_path} column '{col}'")
    return ids


class ArtBenchDataset(Dataset):
    """Wraps a HuggingFace split with optional index subsetting."""

    def __init__(self, hf_split, transform, indices: Optional[list[int]] = None):
        self.ds = hf_split
        self.transform = transform
        self.indices = list(range(len(hf_split))) if indices is None else indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        ex = self.ds[self.indices[idx]]
        x = self.transform(ex["image"])
        y = int(ex["label"])
        return x, y


def build_transform(image_size: int) -> T.Compose:
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
        T.RandomHorizontalFlip(),       # mild augmentation
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # -> [-1, 1]
    ])


def build_loader(
    hf_split, cfg: Config, indices: Optional[list[int]], shuffle: bool
) -> DataLoader:
    ds = ArtBenchDataset(hf_split, build_transform(cfg.image_size), indices)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
    )


def make_cosine_schedule(T: int, s: float = 0.008, device="cpu") -> dict:
    """
    Cosine noise schedule (Nichol & Dhariwal, 2021).
    Smoother alpha_bar decay than the linear schedule, especially beneficial
    at 32x32 resolution.
    """
    steps = T + 1
    t = torch.linspace(0, T, steps, dtype=torch.float64)
    f_t = torch.cos(((t / T) + s) / (1.0 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = (f_t / f_t[0]).float()

    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = betas.clamp(0.0, 0.999)

    alphas_cumprod = alphas_cumprod[1:].to(device)
    betas = betas.to(device)
    alphas = 1.0 - betas
    alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), alphas_cumprod[:-1]])

    return {
        "T": T,
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": alphas_cumprod.sqrt(),
        "sqrt_one_minus_alphas_cumprod": (1.0 - alphas_cumprod).sqrt(),
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "posterior_variance": (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ),
    }



class GaussianDiffusion:
    """
    DDPM forward process + DDPM and DDIM reverse samplers.
    """

    def __init__(self, schedule: dict, device: torch.device):
        self.T = schedule["T"]
        self.device = device
        for key, val in schedule.items():
            if isinstance(val, torch.Tensor):
                setattr(self, key, val.to(device))

    def _gather(self, coef: torch.Tensor, t: torch.Tensor, ndim: int):
        out = coef.gather(0, t)
        return out.view(t.shape[0], *([1] * (ndim - 1)))


    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample x_t from q(x_t | x_0) using the reparameterisation trick."""
        if noise is None:
            noise = torch.randn_like(x0)
        sa = self._gather(self.sqrt_alphas_cumprod, t, x0.ndim)
        sm = self._gather(self.sqrt_one_minus_alphas_cumprod, t, x0.ndim)
        return sa * x0 + sm * noise, noise

    @torch.no_grad()
    def p_sample_ddpm(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_index: int,
        cfg_scale: float = 1.0,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        beta_t = self._gather(self.betas, t, x_t.ndim)
        sm = self._gather(self.sqrt_one_minus_alphas_cumprod, t, x_t.ndim)
        sra = 1.0 / self._gather(self.alphas, t, x_t.ndim).sqrt()

        pred_noise = self._predict_noise(model, x_t, t, cfg_scale, labels)

        mean = sra * (x_t - (beta_t / sm) * pred_noise)
        if t_index == 0:
            return mean
        post_var = self._gather(self.posterior_variance, t, x_t.ndim)
        return mean + post_var.sqrt() * torch.randn_like(x_t)

    @torch.no_grad()
    def p_sample_loop_ddpm(
        self,
        model: nn.Module,
        shape: tuple,
        cfg_scale: float = 1.0,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        model.eval()
        x = torch.randn(shape, device=self.device)
        for i in tqdm(reversed(range(self.T)), desc="DDPM sampling", leave=False, total=self.T):
            t = torch.full((shape[0],), i, dtype=torch.long, device=self.device)
            x = self.p_sample_ddpm(model, x, t, i, cfg_scale, labels)
        return x


    @torch.no_grad()
    def p_sample_loop_ddim(
        self,
        model: nn.Module,
        shape: tuple,
        ddim_steps: int = 100,
        eta: float = 0.0,
        cfg_scale: float = 1.0,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        DDIM deterministic sampler.
        eta=0  -> fully deterministic (recommended for evaluation)
        eta=1  -> recovers stochastic DDPM variance
        """
        model.eval()
        # Build a uniformly-spaced sub-sequence of timesteps
        step_size = self.T // ddim_steps
        timesteps = list(reversed(range(0, self.T, step_size)))  # [T-1, ..., 0]

        x = torch.randn(shape, device=self.device)

        for i, t_val in enumerate(tqdm(timesteps, desc="DDIM sampling", leave=False)):
            t = torch.full((shape[0],), t_val, dtype=torch.long, device=self.device)
            t_prev_val = timesteps[i + 1] if i + 1 < len(timesteps) else -1

            a_bar = self._gather(self.alphas_cumprod, t, x.ndim)
            if t_prev_val >= 0:
                t_prev = torch.full((shape[0],), t_prev_val, dtype=torch.long, device=self.device)
                a_bar_prev = self._gather(self.alphas_cumprod, t_prev, x.ndim)
            else:
                a_bar_prev = torch.ones_like(a_bar)

            pred_noise = self._predict_noise(model, x, t, cfg_scale, labels)

            # Estimate x_0 from current x_t and predicted noise
            x0_pred = (x - (1.0 - a_bar).sqrt() * pred_noise) / a_bar.sqrt()
            x0_pred = x0_pred.clamp(-1.0, 1.0)

            # Compute sigma for DDIM
            sigma = eta * ((1.0 - a_bar_prev) / (1.0 - a_bar) * (1.0 - a_bar / a_bar_prev)).sqrt()

            # Direction pointing to x_t
            dir_xt = (1.0 - a_bar_prev - sigma ** 2).sqrt() * pred_noise

            x = a_bar_prev.sqrt() * x0_pred + dir_xt
            if eta > 0.0 and t_prev_val >= 0:
                x = x + sigma * torch.randn_like(x)

        return x

    # ------------------------------------------------------------------
    # Shared CFG noise prediction
    # ------------------------------------------------------------------

    def _predict_noise(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cfg_scale: float,
        labels: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Classifier-Free Guidance interpolation.
        If cfg_scale == 1.0 or labels is None, returns the conditional (or
        unconditional) prediction without blending.
        """
        if labels is None or cfg_scale <= 1.0:
            return model(x_t, t, labels)

        # One forward pass for conditional, one for unconditional (null class)
        # Batched: concatenate and split to avoid two full forward passes
        null_labels = torch.full_like(labels, model.num_classes)  # null token
        x_double = torch.cat([x_t, x_t], dim=0)
        t_double = torch.cat([t, t], dim=0)
        lbl_double = torch.cat([labels, null_labels], dim=0)

        pred = model(x_double, t_double, lbl_double)
        pred_cond, pred_uncond = pred.chunk(2, dim=0)

        return pred_uncond + cfg_scale * (pred_cond - pred_uncond)


# ---------------------------------------------------------------------------
# U-Net building blocks
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal timestep embedding (Vaswani et al., 2017)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        emb = t[:, None].float() * freqs[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, groups: int = 8):
        super().__init__()
        # Adjust groups to be a divisor of in_ch and out_ch
        groups_in = min(groups, in_ch) if in_ch % groups != 0 else groups
        groups_out = min(groups, out_ch) if out_ch % groups != 0 else groups

        self.norm1 = nn.GroupNorm(groups_in, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups_out, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act = nn.SiLU()
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, out_ch))
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention for spatial feature maps."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        h, _ = self.attn(h, h, h)
        return x + h.permute(0, 2, 1).view(B, C, H, W)


class ArtBenchUNet(nn.Module):
    """
    U-Net noise predictor for 3-channel 32x32 images with optional
    class conditioning for Classifier-Free Guidance.

    Fixed 3-level encoder-decoder matching the proven starter notebook design:

        Encoder:  32x32 (C)  ->  16x16 (2C)  ->  bottleneck at 8x8 (4C)
        Decoder:  8x8  (4C)  ->  16x16 (2C)  ->  32x32 (C)

    Skip connections pass the encoder feature maps directly to the decoder.
    Channel widths: [C, 2C, 4C] where C = model_channels.

    Class conditioning (CFG):
        A learned embedding for each class (+ 1 null token) is added to the
        time embedding before every ResBlock, following Ho & Salimans (2022).
    """

    def __init__(
        self,
        in_channels: int = 3,
        model_channels: int = 128,
        channel_mult: tuple = (1, 2, 4),   # kept for API compatibility, fixed to 3 levels
        num_res_blocks: int = 2,
        num_classes: int = 10,
        use_cfg: bool = True,
        use_attention: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_cfg = use_cfg

        C = model_channels
        t_dim = C * 4

        # Time embedding: sinusoidal -> MLP
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(C),
            nn.Linear(C, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )

        # Class embedding: num_classes regular + 1 null token for CFG dropout
        n_emb = num_classes + 1 if use_cfg else num_classes
        self.class_embed = nn.Embedding(n_emb, t_dim)

        # ---- Encoder ----
        self.init_conv = nn.Conv2d(in_channels, C, 3, padding=1)

        # Level 0: 32x32, C channels
        self.down0 = nn.ModuleList([ResBlock(C, C, t_dim) for _ in range(num_res_blocks)])
        self.pool0 = nn.Conv2d(C, C, 3, stride=2, padding=1)          # 32 -> 16

        # Level 1: 16x16, 2C channels
        self.down1 = nn.ModuleList(
            [ResBlock(C if i == 0 else C * 2, C * 2, t_dim) for i in range(num_res_blocks)]
        )
        self.pool1 = nn.Conv2d(C * 2, C * 2, 3, stride=2, padding=1)  # 16 -> 8

        # ---- Bottleneck: 8x8, 4C channels ----
        self.mid0 = ResBlock(C * 2, C * 4, t_dim)
        self.mid_attn = SelfAttention(C * 4) if use_attention else nn.Identity()
        self.mid1 = ResBlock(C * 4, C * 4, t_dim)

        # ---- Decoder ----
        # Level 1 up: 8 -> 16, skip from down1 output (2C channels)
        self.up1_conv = nn.ConvTranspose2d(C * 4, C * 2, 4, stride=2, padding=1)
        self.up1 = nn.ModuleList(
            [ResBlock(C * 4 if i == 0 else C * 2, C * 2, t_dim) for i in range(num_res_blocks)]
        )   # first block: cat(2C upsampled + 2C skip) = 4C in

        # Level 0 up: 16 -> 32, skip from down0 output (C channels)
        self.up0_conv = nn.ConvTranspose2d(C * 2, C, 4, stride=2, padding=1)
        self.up0 = nn.ModuleList(
            [ResBlock(C * 2 if i == 0 else C, C, t_dim) for i in range(num_res_blocks)]
        )   # first block: cat(C upsampled + C skip) = 2C in

        # ---- Output ----
        self.out_norm = nn.GroupNorm(8, C)
        self.out_conv = nn.Conv2d(C, in_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:      (B, 3, 32, 32) noisy image in [-1, 1]
            t:      (B,) integer timestep indices
            labels: (B,) class labels in [0, num_classes-1] or num_classes for null token
        Returns:
            (B, 3, 32, 32) predicted noise
        """
        t_emb = self.time_embed(t)   # (B, t_dim)

        if labels is not None and self.use_cfg:
            t_emb = t_emb + self.class_embed(labels)

        # Encoder
        h = self.init_conv(x)              # (B, C, 32, 32)
        for blk in self.down0:
            h = blk(h, t_emb)
        skip0 = h                          # (B, C, 32, 32)
        h = self.pool0(h)                  # (B, C, 16, 16)

        for blk in self.down1:
            h = blk(h, t_emb)
        skip1 = h                          # (B, 2C, 16, 16)
        h = self.pool1(h)                  # (B, 2C, 8, 8)

        # Bottleneck
        h = self.mid0(h, t_emb)            # (B, 4C, 8, 8)
        if isinstance(self.mid_attn, SelfAttention):
            h = self.mid_attn(h)
        h = self.mid1(h, t_emb)            # (B, 4C, 8, 8)

        # Decoder level 1: 8 -> 16
        h = self.up1_conv(h)               # (B, 2C, 16, 16)
        h = torch.cat([h, skip1], dim=1)   # (B, 4C, 16, 16)
        for blk in self.up1:
            h = blk(h, t_emb)             # (B, 2C, 16, 16)

        # Decoder level 0: 16 -> 32
        h = self.up0_conv(h)               # (B, C, 32, 32)
        h = torch.cat([h, skip0], dim=1)   # (B, 2C, 32, 32)
        for blk in self.up0:
            h = blk(h, t_emb)             # (B, C, 32, 32)

        h = F.silu(self.out_norm(h))
        return self.out_conv(h)            # (B, 3, 32, 32)


class EMA: #usei claude professor peço desculpa
    """
    Exponential moving average of model parameters.
    EMA weights are used exclusively at evaluation / sampling time.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_p, p in zip(self.shadow.parameters(), model.parameters()):
            ema_p.copy_(self.decay * ema_p + (1.0 - self.decay) * p)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, sd):
        self.shadow.load_state_dict(sd)


def train(
    model, loader, diffusion, cfg, device, run_name,
    start_epoch=1,          # <-- novo
    ema=None,
    optimiser_state=None,# <-- novo parâmetro              # <-- novo, para passar EMA já inicializado
) -> list[float]:
    
    model_dir = Path(cfg.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    history_dir = Path(cfg.history_dir)
    history_dir.mkdir(parents=True, exist_ok=True)

    model.to(device)
    if ema is None:
        ema = EMA(model, cfg.ema_decay) if cfg.use_ema else None

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    if optimiser_state is not None:
        optimiser.load_state_dict(optimiser_state)

    def lr_lambda(epoch):
        if epoch < cfg.warmup_epochs:
            return (epoch + 1) / max(cfg.warmup_epochs, 1)
        progress = (epoch - cfg.warmup_epochs) / max(cfg.epochs - cfg.warmup_epochs, 1)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(start_epoch - 1):
            scheduler.step()

    history = []
    for epoch in range(start_epoch, cfg.epochs + 1):
        # ... igual ao que já tens
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for x, y in tqdm(loader, desc=f"Epoch {epoch:04d}/{cfg.epochs}", leave=False):
            x = x.to(device)
            y = y.to(device)
            B = x.shape[0]

            # Randomly drop class labels for CFG training
            if cfg.use_cfg and cfg.p_uncond > 0.0:
                drop_mask = torch.rand(B, device=device) < cfg.p_uncond
                y = y.masked_fill(drop_mask, model.num_classes)  # null token

            t = torch.randint(0, diffusion.T, (B,), device=device, dtype=torch.long)
            x_t, noise = diffusion.q_sample(x, t)

            noise_pred = model(x_t, t, y if cfg.use_cfg else None)
            loss = F.mse_loss(noise_pred, noise)

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimiser.step()

            if ema is not None:
                ema.update(model)

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        avg = epoch_loss / max(n_batches, 1)
        history.append(avg)
        lr_now = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:04d}/{cfg.epochs}  loss={avg:.5f}  lr={lr_now:.2e}")

        # Checkpoint every 50 epochs
        if epoch % 25 == 0 or epoch == cfg.epochs:
            ckpt = {
                "model": model.state_dict(),
                "ema": ema.state_dict() if ema is not None else None,
                "epoch": epoch,
                "history": history,
                "config": cfg.__dict__,
            }
            ckpt_path = model_dir / f"{run_name}_ep{epoch:04d}.pt"
            torch.save(ckpt, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    # Save final
    final_path = model_dir / f"{run_name}_final.pt"
    torch.save({
        "model": model.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "optimiser": optimiser.state_dict(),
        "epoch": cfg.epochs,
        "history": history,
        "config": cfg.__dict__,
    }, final_path)
    (history_dir / f"{run_name}.json").write_text(json.dumps(history, indent=2))
    print(f"Training complete. Model saved to {final_path}")

    return history


@torch.no_grad()
def generate_samples(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    cfg: Config,
    device: torch.device,
    n: int = 5000,
    batch_size: int = 128,
    use_ddim: bool = True,
    seed: int = 0,
) -> torch.Tensor:
    """
    Generate `n` samples and return as float32 tensor in [0, 1] (N, 3, H, W).
    Uses EMA model if available (passed as `model`).
    Class labels are sampled uniformly from all classes.
    """
    set_seed(seed)
    model.eval()
    all_samples = []

    for start in range(0, n, batch_size):
        b = min(batch_size, n - start)
        labels = None
        if cfg.use_cfg:
            labels = torch.randint(0, cfg.num_classes, (b,), device=device)

        shape = (b, 3, cfg.image_size, cfg.image_size)
        if use_ddim:
            x = diffusion.p_sample_loop_ddim(
                model, shape,
                ddim_steps=cfg.ddim_steps,
                eta=cfg.ddim_eta,
                cfg_scale=cfg.cfg_scale,
                labels=labels,
            )
        else:
            x = diffusion.p_sample_loop_ddpm(
                model, shape,
                cfg_scale=cfg.cfg_scale,
                labels=labels,
            )

        # Denormalize [-1,1] -> [0,1]
        x = (x.clamp(-1.0, 1.0) + 1.0) * 0.5
        all_samples.append(x.cpu())

    return torch.cat(all_samples, dim=0)[:n]


def get_real_images_uint8(
    hf_split, n: int, seed: int, image_size: int
) -> torch.Tensor:
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(hf_split), size=n, replace=False).tolist()
    tf = T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(image_size),
        T.ToTensor(),
    ])
    imgs = [tf(hf_split[int(i)]["image"]) for i in indices]
    return (torch.stack(imgs) * 255).to(torch.uint8)


def compute_fid_kid(
    model, diffusion, hf_split, cfg, device,
    n_samples=5000, kid_subsets=50, kid_subset_size=100,
    gen_batch_size=128, seed=0,
) -> dict:
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance
    except ImportError:
        raise RuntimeError("pip install torchmetrics[image]")

    fake = generate_samples(model, diffusion, cfg, device, n=n_samples,
                            batch_size=gen_batch_size, seed=seed)
    fake_u8 = (fake * 255).to(torch.uint8)
    real_u8 = get_real_images_uint8(hf_split, n_samples, seed, cfg.image_size)

    fid_m = FrechetInceptionDistance(feature=2048).to(device)
    kid_m = KernelInceptionDistance(
        feature=2048, subsets=kid_subsets, subset_size=kid_subset_size
    ).to(device)

    # Feed in batches to avoid OOM on Inception upsample
    inception_batch = 64
    for i in range(0, n_samples, inception_batch):
        r = real_u8[i:i+inception_batch].to(device)
        f = fake_u8[i:i+inception_batch].to(device)
        fid_m.update(r, real=True)
        fid_m.update(f, real=False)
        kid_m.update(r, real=True)
        kid_m.update(f, real=False)

    fid_val = fid_m.compute().item()
    kid_mean, kid_std = kid_m.compute()

    return {"fid": fid_val, "kid_mean": kid_mean.item(), "kid_std": kid_std.item()}

def evaluate_over_seeds(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    hf_split,
    cfg: Config,
    device: torch.device,
    n_seeds: int = 10,
    n_samples: int = 5000,
    gen_batch_size: int = 128,
) -> dict:
    fids, k_means, k_stds = [], [], []

    for seed in range(n_seeds):
        print(f"\n--- Seed {seed + 1}/{n_seeds} ---")
        r = compute_fid_kid(
            model, diffusion, hf_split, cfg, device,
            n_samples=n_samples, gen_batch_size=gen_batch_size, seed=seed,
        )
        fids.append(r["fid"])
        k_means.append(r["kid_mean"])
        k_stds.append(r["kid_std"])
        print(f"  FID={r['fid']:.2f}  KID={r['kid_mean']:.5f}+/-{r['kid_std']:.5f}")

    summary = {
        "fid_mean": float(np.mean(fids)),
        "fid_std": float(np.std(fids)),
        "kid_mean_avg": float(np.mean(k_means)),
        "kid_std_avg": float(np.mean(k_stds)),
    }
    print("\n====== Final evaluation ======")
    print(f"FID : {summary['fid_mean']:.2f} +/- {summary['fid_std']:.2f}")
    print(f"KID : {summary['kid_mean_avg']:.5f} +/- {summary['kid_std_avg']:.5f}")
    return summary


def save_sample_grid(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    cfg: Config,
    device: torch.device,
    n: int = 64,
    path: Optional[Path] = None,
) -> None:
    samples = generate_samples(model, diffusion, cfg, device, n=n, batch_size=n)
    grid = make_grid(samples, nrow=8, padding=2)
    path = path or Path(cfg.sample_dir) / f"{cfg.run_name}_{n}_samples.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, path)
    print(f"Sample grid saved: {path}")


def show_noising_trajectory(x0: torch.Tensor, diffusion: GaussianDiffusion) -> None:
    import matplotlib.pyplot as plt
    steps = [0, 100, 250, 500, 750, 999]
    fig, axes = plt.subplots(1, len(steps), figsize=(len(steps) * 2.2, 2.5))
    for ax, step in zip(axes, steps):
        t = torch.tensor([step], dtype=torch.long, device=diffusion.device)
        x_t, _ = diffusion.q_sample(x0.to(diffusion.device), t)
        img = ((x_t[0].cpu() + 1.0) * 0.5).permute(1, 2, 0).clamp(0, 1).numpy()
        ax.imshow(img)
        ax.set_title(f"t={step}")
        ax.axis("off")
    plt.suptitle("Forward diffusion trajectory")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="DDPM for ArtBench-10")
    parser.add_argument("--mode", choices=["train", "eval", "sample"], default="train")
    parser.add_argument("--config", choices=list(CONFIGS.keys()), default="medium")
    parser.add_argument("--full_dataset", action="store_true", help="Train on full 50k instead of 20 percent subset")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt checkpoint for eval/sample mode")
    parser.add_argument("--epochs", type=int, default=None,help="Override number of epochs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_eval_seeds", type=int, default=10)
    parser.add_argument("--project_root", type=str, default="../DATA")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint .pt para continuar treino")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    cfg = CONFIGS[args.config]
    cfg.project_root = args.project_root
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if not cfg.run_name:
        cfg.run_name = cfg.derive_run_name()

    # Load dataset
    kaggle_root = Path(cfg.project_root) / "ArtBench-10"
    hf_ds = load_artbench_splits(kaggle_root)
    train_hf = hf_ds["train"]

    # Build loader
    if args.full_dataset:
        indices = None
        print(f"Using full training set ({len(train_hf)} images)")
        cfg.run_name += "_full"
    else:
        subset_csv = Path(cfg.subset_csv)
        indices = load_subset_ids(subset_csv)
        print(f"Using 20% subset ({len(indices)} images)")

    # Build model and diffusion
    model = ArtBenchUNet(
        in_channels=3,
        model_channels=cfg.model_channels,
        channel_mult=cfg.channel_mult,
        num_res_blocks=cfg.num_res_blocks,
        num_classes=cfg.num_classes,
        use_cfg=cfg.use_cfg,
        use_attention=cfg.use_attention,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params / 1e6:.1f}M")

    schedule = make_cosine_schedule(cfg.T, s=cfg.cosine_s, device=device)
    diffusion = GaussianDiffusion(schedule, device)

    if args.mode == "train":
        loader = build_loader(train_hf, cfg, indices, shuffle=True)

        start_epoch = 1
        resume_ema = None
        history_so_far = []

        if args.resume:
            ckpt = torch.load(args.resume, map_location=device)
            model.load_state_dict(ckpt["model"])
            start_epoch = ckpt["epoch"] + 1
            history_so_far = ckpt.get("history", [])
            if ckpt.get("ema") is not None and cfg.use_ema:
                resume_ema = EMA(model, cfg.ema_decay)
                resume_ema.load_state_dict(ckpt["ema"])
                resume_ema.shadow.to(device)
            resume_optimiser_state = ckpt.get("optimiser", None)  # <-- adiciona isto
            print(f"Resumed from epoch {ckpt['epoch']} -> continuing to epoch {cfg.epochs}")
        history_new = train(
            model, loader, diffusion, cfg, device, cfg.run_name,
            start_epoch=start_epoch,
            ema=resume_ema,
            optimiser_state=resume_optimiser_state
        )
        history = history_so_far + history_new

        # Save sample grid after training
        ckpt = torch.load(
            Path(cfg.model_dir) / f"{cfg.run_name}_final.pt", map_location=device
        )
        eval_model = deepcopy(model)
        # Prefer EMA weights for sampling
        if ckpt["ema"] is not None:
            eval_model.load_state_dict(ckpt["ema"])
        else:
            eval_model.load_state_dict(ckpt["model"])
        eval_model.to(device)
        save_sample_grid(eval_model, diffusion, cfg, device)

    elif args.mode == "eval":
        if args.checkpoint is None:
            raise ValueError("--checkpoint required for eval mode")
        ckpt = torch.load(args.checkpoint, map_location=device)
        eval_model = deepcopy(model)
        if ckpt.get("ema") is not None:
            eval_model.load_state_dict(ckpt["ema"])
            print("Loaded EMA weights for evaluation")
        else:
            eval_model.load_state_dict(ckpt["model"])
        eval_model.to(device)

        summary = evaluate_over_seeds(
            eval_model, diffusion, train_hf, cfg, device,
            n_seeds=args.n_eval_seeds,
        )
        out_path = Path(cfg.history_dir) / f"{cfg.run_name}_eval.json"
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"Evaluation results saved to {out_path}")

    elif args.mode == "sample":
        if args.checkpoint is None:
            raise ValueError("--checkpoint required for sample mode")
        ckpt = torch.load(args.checkpoint, map_location=device)
        eval_model = deepcopy(model)
        if ckpt.get("ema") is not None:
            eval_model.load_state_dict(ckpt["ema"])
        else:
            eval_model.load_state_dict(ckpt["model"])
        eval_model.to(device)
        save_sample_grid(eval_model, diffusion, cfg, device, n=512)


if __name__ == "__main__":
    main()