from __future__ import annotations

import sys
import csv
import json
import random
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

PROJECT_ROOT = Path("../DATA")
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
KAGGLE_ROOT = PROJECT_ROOT / "ArtBench-10"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

IMAGE_SIZE = 32
BATCH_SIZE = 64
NUM_WORKERS = 0
LATENT_DIM = 128
EPOCHS = 300
LR = 1e-3
BETA = 0.1
SPATIAL_FLAT = 256 * 4 * 4  

USE_SAVED_MODELS_IF_AVAILABLE = False

TRAINING_CSV_PATH = Path("../DATA/training_20_percent.csv")
INDEX_COLUMN = "train_id_original"

MODEL_DIR   = Path("./models")
HISTORY_DIR = Path("./histories")
SAMPLES_DIR = Path("./Samples_per_epoch/low_beta(0.1)_vae_latent256_samples")

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")

device = get_device()

def safe_num_workers(requested: int) -> int:
    if "ipykernel" in sys.modules and int(requested) > 0:
        print("Notebook kernel detected: forcing num_workers=0 for DataLoader stability.")
        return 0
    return int(requested)


class HFDatasetTorch(Dataset):
    def __init__(self, hf_split, transform=None, indices=None):
        self.ds        = hf_split
        self.transform = transform
        self.indices   = list(range(len(hf_split))) if indices is None else list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        ex = self.ds[real_idx]
        img = ex["image"]
        y = int(ex["label"])
        x = self.transform(img) if self.transform else img
        return x, y, real_idx


def make_subset_indices(n_total: int, fraction: float, seed: int = 42):
    n_keep = max(1, int(round(n_total * fraction)))
    g = np.random.RandomState(seed)
    idx = np.arange(n_total)
    g.shuffle(idx)
    return idx[:n_keep].tolist()


def load_ids_from_training_csv(
    csv_path: Path, index_column: str = "train_id_original"
) -> list[int]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"training.csv not found: {csv_path}\n"
            "Generate it first with scripts/generate_training_csv.py"
        )
    ids = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if index_column not in (r.fieldnames or []):
            raise ValueError(
                f"Column {index_column!r} not present in {csv_path}. "
                f"Available: {r.fieldnames}"
            )
        for row in r:
            v = str(row.get(index_column, "")).strip()
            if v == "":
                continue
            ids.append(int(v))
    if len(ids) == 0:
        raise ValueError(f"No ids found in {csv_path} column {index_column!r}")
    return ids


def show_batch_grid(loader, class_names, n_images=36, nrow=6, title="Sample Grid"):
    x, y, _ = next(iter(loader))
    x = x[:n_images]
    y = y[:n_images]
    grid = make_grid(x, nrow=nrow, padding=2)
    np_img = grid.permute(1, 2, 0).cpu().numpy()
    plt.figure(figsize=(8, 8))
    plt.imshow(np_img)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig("batch_grid.png")
    plt.close()
    labels_str = [class_names[int(v)] for v in y]
    print("Labels:", labels_str)


def _artifact_stem(run_name: str) -> str:
    stem = "".join(ch.lower() if ch.isalnum() else "_" for ch in run_name)
    stem = "_".join(part for part in stem.split("_") if part)
    return stem or "model"


def _artifact_paths(run_name: str):
    stem = _artifact_stem(run_name)
    return MODEL_DIR / f"{stem}.pt", HISTORY_DIR / f"{stem}.json"


def fit_or_load_model(
    model, run_name, train_fn,
    load_if_available=USE_SAVED_MODELS_IF_AVAILABLE,
    **train_kwargs,
):
    model_path, history_path = _artifact_paths(run_name)

    if load_if_available and model_path.exists():
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        if history_path.exists():
            with history_path.open("r", encoding="utf-8") as f:
                history = json.load(f)
        else:
            history = []
        print(f"Loaded {run_name} model from {model_path}")
        return history

    history = train_fn(model=model, **train_kwargs)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as f:
        torch.save(model.state_dict(), f)
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved {run_name} model to {model_path} and history to {history_path}")
    return history


class ConvVAE(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc_mu = nn.Linear(SPATIAL_FLAT, latent_dim)
        self.fc_logvar = nn.Linear(SPATIAL_FLAT, latent_dim)
        self.dec_fc = nn.Linear(latent_dim, SPATIAL_FLAT)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z):
        h = self.dec_fc(z)
        h = h.view(h.size(0), 256, 4, 4)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        xhat = self.decode(z)
        return xhat, mu, logvar

    @torch.no_grad()
    def sample(self, n: int, device: torch.device):
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode(z)


def vae_loss(xhat, x, mu, logvar, beta: float = 1.0):
    B = x.size(0)
    recon = F.mse_loss(xhat, x, reduction="sum") / B
    kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp()) / B
    total = recon + beta * kl
    return total, recon, kl


def train_vae(model, loader, optimizer, epochs=EPOCHS, beta_max=BETA, scheduler=None):
    from tqdm import tqdm
    model.train()
    history = []

    for ep in range(1, epochs + 1):
        beta = min(beta_max, beta_max * (ep / (epochs * 0.6)))
        total_loss = recon_loss = kl_loss = 0.0

        for x, _y, _idx in tqdm(loader, desc=f"Epoch {ep}/{epochs}", leave=False):
            x = x.to(device)
            optimizer.zero_grad()
            xhat, mu, logvar = model(x)
            loss, recon, kl = vae_loss(xhat, x, mu, logvar, beta=beta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            n = x.size(0)
            total_loss += loss.item() * n
            recon_loss += recon.item() * n
            kl_loss += kl.item() * n

        if scheduler is not None:
            scheduler.step()

        N = len(loader.dataset)
        entry = {
            "epoch": ep,
            "train_loss": total_loss / N,
            "train_recon": recon_loss / N,
            "train_kl": kl_loss    / N,
        }
        history.append(entry)
        print(
            f"Epoch {ep:3d}/{epochs} | "
            f"loss={entry['train_loss']:.4f}  "
            f"recon={entry['train_recon']:.4f}  "
            f"kl={entry['train_kl']:.4f}"
        )

        if ep % 10 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                z = torch.randn(64, model.latent_dim, device=device)
                samples = model.decode(z)
            SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
            save_image(samples, SAMPLES_DIR / f"samples_ep{ep:04d}.png", nrow=8)
            model.train()

    return history


def evaluate_vae(model, loader, beta=BETA):
    model.eval()
    total_loss = recon_loss = kl_loss = mse_sum = mae_sum = 0.0
    n_samples = 0
    last_x = None

    with torch.no_grad():
        for x, _y, _idx in loader:
            x = x.to(device)
            xhat, mu, logvar = model(x)
            B = x.size(0)

            loss, recon, kl = vae_loss(xhat, x, mu, logvar, beta=beta)

            total_loss += loss.item()  * B
            recon_loss += recon.item() * B
            kl_loss += kl.item()    * B
            mse_sum += F.mse_loss(xhat, x, reduction="sum").item()
            mae_sum += (xhat - x).abs().sum().item()
            n_samples += B
            last_x = x

    numel = last_x[0].numel()
    return {
        "loss": total_loss / n_samples,
        "recon_mse": recon_loss / n_samples,
        "kl": kl_loss / n_samples,
        "mse": mse_sum / (n_samples * numel),
        "mae": mae_sum / (n_samples * numel),
    }


def show_reconstructions(model, loader, n=8, save_path="reconstructions_standard_beta.png"):
    model.eval()
    x, _y, _idx = next(iter(loader))
    x = x[:n].to(device)

    with torch.no_grad():
        xhat, _, _ = model(x)

    comparison = torch.cat([x.cpu(), xhat.cpu()], dim=0)
    grid = make_grid(comparison, nrow=n, padding=2)

    plt.figure(figsize=(n * 1.5, 4))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis("off")
    plt.title("Original vs reconstructions")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved reconstructions to {save_path}")


def show_prior_samples(model, n=64, nrow=8, save_path="prior_samples_standard_beta.png"):
    model.eval()
    with torch.no_grad():
        z = torch.randn(n, model.latent_dim, device=device)
        samples = model.decode(z)

    grid = make_grid(samples.cpu(), nrow=nrow, padding=2)

    plt.figure(figsize=(nrow * 1.5, (n // nrow) * 1.5))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis("off")
    plt.title("Samples from N(0,I)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved prior samples to {save_path}")

def make_generate_fn(model: ConvVAE, device: torch.device):
    @torch.no_grad()
    def generate_fn(n: int, batch_size: int, seed: int) -> torch.Tensor:
        torch.manual_seed(seed)
        model.eval()
        chunks = []
        for start in range(0, n, batch_size):
            bs = min(batch_size, n - start)
            imgs = model.sample(bs, device).cpu()
            chunks.append(imgs)
        return torch.cat(chunks, dim=0)   

    return generate_fn


def parse_args():
    p = argparse.ArgumentParser(description="beta-VAE training and evaluation")
    p.add_argument("--mode", choices=["train", "eval_recon", "eval_fid", "eval_fid_seeds"], default="train")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--load", action="store_true")
    p.add_argument("--latent_dim", type=int,   default=LATENT_DIM)
    p.add_argument("--epochs", type=int,   default=EPOCHS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--beta", type=float, default=BETA)
    p.add_argument("--batch_size", type=int,   default=BATCH_SIZE)
    p.add_argument("--image_size", type=int,   default=IMAGE_SIZE)
    p.add_argument("--num_workers",type=int,   default=NUM_WORKERS)
    p.add_argument("--n_samples", type=int, default=5000)
    p.add_argument("--gen_batch_size", type=int, default=128)
    p.add_argument("--n_seeds", type=int, default=10)
    p.add_argument("--seed",type=int, default=0)
    p.add_argument("--training_csv", type=str, default=str(TRAINING_CSV_PATH))
    p.add_argument("--run_name", type=str, default="artbench_beta_VAE")
    p.add_argument("--kaggle_root", type=str, default=str(KAGGLE_ROOT))
    p.add_argument("--scripts_dir", type=str, default=str(SCRIPTS_DIR))
    return p.parse_args()


def main():
    args = parse_args()

    kaggle_root = Path(args.kaggle_root)
    scripts_dir = Path(args.scripts_dir)
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from artbench_local_dataset import load_kaggle_artbench10_splits
    hf_ds    = load_kaggle_artbench10_splits(kaggle_root)
    train_hf = hf_ds["train"]
    print("Train size:", len(train_hf))

    label_feature = train_hf.features["label"]
    class_names   = list(label_feature.names)
    num_classes   = len(class_names)
    print("Num classes:", num_classes)

    transform = T.Compose([
        T.Resize(args.image_size, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(args.image_size),
        T.ToTensor(),
    ])

    train_ids = load_ids_from_training_csv(Path(args.training_csv), INDEX_COLUMN)
    train_ds  = HFDatasetTorch(train_hf, transform=transform, indices=train_ids)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=safe_num_workers(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    print("Subset train dataset length:", len(train_ds))

    
    vae = ConvVAE(latent_dim=args.latent_dim).to(device)
    print(f"Total parameters: {sum(p.numel() for p in vae.parameters()):,}")

    
    if args.checkpoint is not None:
        ckpt_path = Path(args.checkpoint)
        state_dict = torch.load(ckpt_path, map_location=device)
        vae.load_state_dict(state_dict)
        vae.to(device)
        print(f"Loaded checkpoint: {ckpt_path}")

    
    if args.mode == "train":
        optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-5
        )
        history = fit_or_load_model(
            model=vae,
            run_name=args.run_name,
            train_fn=train_vae,
            load_if_available=args.load,
            loader=train_loader,
            optimizer=optimizer,
            epochs=args.epochs,
            beta_max=args.beta,
            scheduler=scheduler,
        )
        metrics = evaluate_vae(vae, train_loader, beta=args.beta)
        print("Reconstruction metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.6f}")
        show_reconstructions(vae, train_loader)
        show_prior_samples(vae)
    elif args.mode == "eval_recon":
        metrics = evaluate_vae(vae, train_loader, beta=args.beta)
        print("Reconstruction metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.6f}")
        show_reconstructions(vae, train_loader)
        show_prior_samples(vae)


if __name__ == "__main__":
    main()