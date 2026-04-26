import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

PROJECT_ROOT = Path('../DATA')
SCRIPTS_DIR  = PROJECT_ROOT / 'scripts'
KAGGLE_ROOT  = PROJECT_ROOT / 'ArtBench-10'
SAMPLES_DIR  = Path('./samples/DCGAN_LS_half_lr_ngf256')

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

SEED = 42
IMAGE_SIZE = 32
BATCH_SIZE = 64
NUM_WORKERS = 0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

transform = T.Compose([
    T.Resize(IMAGE_SIZE, interpolation=T.InterpolationMode.BILINEAR),
    T.CenterCrop(IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  
])

class HFDatasetTorch(Dataset):
    def __init__(self, hf_split, transform=None, indices=None):
        self.ds        = hf_split
        self.transform = transform
        self.indices   = list(range(len(hf_split))) if indices is None else list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        ex  = self.ds[real_idx]
        img = ex['image']
        y   = int(ex['label'])
        x   = self.transform(img) if self.transform else img
        return x, y, real_idx


def make_subset_indices(n_total, fraction, seed = 42):
    n_keep = max(1, int(round(n_total * fraction)))
    g   = np.random.RandomState(seed)
    idx = np.arange(n_total)
    g.shuffle(idx)
    return idx[:n_keep].tolist()


def load_ids_from_training_csv(csv_path, index_column = 'train_id_original'):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f'training.csv not found: {csv_path}')
    ids = []
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        r = csv.DictReader(f)
        if index_column not in (r.fieldnames or []):
            raise ValueError(f'Column {index_column!r} not in {csv_path}')
        for row in r:
            v = str(row.get(index_column, '')).strip()
            if v:
                ids.append(int(v))
    if not ids:
        raise ValueError(f'No ids found in {csv_path}')
    return ids


class DCGenerator(nn.Module):
    def __init__(self, latent_dim=100, image_channels=3, ngf=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True),
            nn.ConvTranspose2d(ngf, image_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        z = z.view(z.size(0), self.latent_dim, 1, 1)
        return self.net(z)


class DCDiscriminator(nn.Module):
    def __init__(self, image_channels=3, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(image_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.net(x).view(-1, 1)


def init_dcgan_weights(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train_gan(generator, discriminator, loader, latent_dim,epochs=300, lr=2e-4, device='cpu', samples_dir=SAMPLES_DIR):
    samples_dir = Path(samples_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.BCEWithLogitsLoss()
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr * 0.5, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr * 0.5, betas=(0.5, 0.999))

    history = {'g_loss': [], 'd_loss': []}
    generator.train()
    discriminator.train()

    for epoch in range(epochs):
        g_running = d_running = 0.0
        n_batches = 0

        for batch in tqdm(loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False):
            real = batch[0].to(device)
            bs   = real.size(0)

            real_targets = torch.ones(bs, 1, device=device)  * 0.9
            fake_targets = torch.zeros(bs, 1, device=device)

            # Discriminator
            opt_d.zero_grad()
            d_loss = criterion(discriminator(real), real_targets)
            z = torch.randn(bs, latent_dim, device=device)
            fake = generator(z)
            d_loss += criterion(discriminator(fake.detach()), fake_targets)
            d_loss.backward()
            opt_d.step()

            # Generator
            opt_g.zero_grad()
            z = torch.randn(bs, latent_dim, device=device)
            fake = generator(z)
            g_loss = criterion(discriminator(fake), real_targets)
            g_loss.backward()
            opt_g.step()

            g_running += g_loss.item()
            d_running += d_loss.item()
            n_batches += 1

        history['g_loss'].append(g_running / max(n_batches, 1))
        history['d_loss'].append(d_running / max(n_batches, 1))
        print(f"Epoch {epoch+1:03d}/{epochs} | "
              f"D loss: {history['d_loss'][-1]:.4f} | "
              f"G loss: {history['g_loss'][-1]:.4f}")

        if epoch % 50 == 0 or epoch == epochs - 1:
            generator.eval()
            with torch.no_grad():
                z = torch.randn(64, latent_dim, device=device)
                samples = generator(z)
                samples = (samples * 0.5 + 0.5).clamp(0, 1)
            save_image(samples, samples_dir / f'samples_ep{epoch:04d}.png', nrow=8)
            generator.train()

    return history


def save_checkpoint(generator, discriminator, history, checkpoint_path,latent_dim, channels, image_size):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'history': history,
        'config': {
            'latent_dim': latent_dim,
            'channels': channels,
            'image_size': image_size,
        },
    }, checkpoint_path)
    print('Saved checkpoint to', checkpoint_path)


def load_dcgan_generator_for_inference(checkpoint_path, device=None):
    if device is None:
        device = get_device()
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt['config']
    gen = DCGenerator(latent_dim=cfg['latent_dim'], image_channels=cfg['channels']).to(device)
    gen.load_state_dict(ckpt['generator'])
    gen.eval()
    return gen, cfg, ckpt.get('history', None)


@torch.no_grad()
def generate_images(generator, latent_dim, n_samples, device, seed):
    torch.manual_seed(seed)
    z = torch.randn(n_samples, latent_dim, device=device)
    generator.eval()
    return generator(z).cpu()


def show_image_grid(images, title=None, nrow=8, figsize=(8, 8), cmap='gray'):
    images = images.detach().cpu().float()
    images = (images * 0.5 + 0.5).clamp(0, 1)
    grid = make_grid(images, nrow=nrow)
    np_grid = grid.permute(1, 2, 0).numpy()
    plt.figure(figsize=figsize)
    if np_grid.shape[-1] == 1:
        plt.imshow(np_grid[..., 0], cmap=cmap)
    else:
        plt.imshow(np_grid)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_gan_losses(history, title='GAN losses'):
    plt.figure(figsize=(7, 4))
    plt.plot(history['d_loss'], label='Discriminator loss')
    plt.plot(history['g_loss'], label='Generator loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device('mps')
    return torch.device('cpu')


def main():
    parser = argparse.ArgumentParser(description='DCGAN for ArtBench-10')
    parser.add_argument('--mode', choices=['train', 'eval', 'plot'], default='train')
    parser.add_argument('--checkpoint', type=str, default='models/artBenchDCGAN_LS_halfLR_ngf256.pt')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--csv', type=str, default='../DATA/training_20_percent.csv')
    args = parser.parse_args()

    device = get_device()
    print(f'Device: {device}')

    if args.mode == 'plot':
        _, _, history = load_dcgan_generator_for_inference(args.checkpoint, device)
        plot_gan_losses(history, title='DCGAN ArtBench-10 — Training Losses')
        return

    from artbench_local_dataset import load_kaggle_artbench10_splits
    hf_ds = load_kaggle_artbench10_splits(KAGGLE_ROOT)
    train_hf = hf_ds['train']

    train_ids = load_ids_from_training_csv(Path(args.csv))
    train_ds = HFDatasetTorch(train_hf, transform=transform, indices=train_ids)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

    if args.mode == 'train':
        latent_dim = 100
        channels = 3
        generator = DCGenerator(latent_dim, channels,ngf=256).to(device)
        discriminator = DCDiscriminator(channels,ndf=256).to(device)
        generator.apply(init_dcgan_weights)
        discriminator.apply(init_dcgan_weights)

        history = train_gan(generator, discriminator, train_loader, latent_dim, epochs=args.epochs, lr=args.lr, device=device)

        save_checkpoint(generator, discriminator, history, args.checkpoint, latent_dim, channels, IMAGE_SIZE)
        plot_gan_losses(history)

    elif args.mode == 'eval':
        from metrics import run_evaluation
        generator, cfg, _ = load_dcgan_generator_for_inference(args.checkpoint, device)

        test_hf = hf_ds['test']
        test_ds = HFDatasetTorch(test_hf, transform=transform)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

        evaluation_config = {
            'fid_kid_samples':    5000,
            'num_runs':             10,
            'subset_size':         100,
            'num_subsets':          50,
            'feature_batch_size':   32,
            'generation_seed':     123,
        }

        results = run_evaluation(
            generator = generator,
            latent_dim = cfg['latent_dim'],
            ref_loader = test_loader,
            device = device,
            cfg = evaluation_config,
            generate_fn = generate_images,
        )
        print(results)


if __name__ == '__main__':
    main()