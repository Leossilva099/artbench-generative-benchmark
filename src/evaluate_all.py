"""
evaluate_all.py
===============
Avalia GAN, Diffusion e VAE com o mesmo protocolo (metrics.py).
Resultados guardados em histories/evaluation_results.json
"""

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path('../DATA')
SCRIPTS_DIR  = PROJECT_ROOT / 'scripts'
sys.path.insert(0, str(SCRIPTS_DIR))

# ── Imports locais ────────────────────────────────────────────────────────────
from artbench_local_dataset import load_kaggle_artbench10_splits
from metrics import run_evaluation
from GAN import (
    load_dcgan_generator_for_inference,
    HFDatasetTorch,
    generate_images,
    DCGenerator,
)
from diffusion import (
    ArtBenchUNet, GaussianDiffusion, CONFIGS,
    make_cosine_schedule, generate_samples,
)
from beta_VAE import ConvVAE, make_generate_fn

# ── Device ────────────────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

device = get_device()
print(f"Device: {device}")

# ── Dataset ───────────────────────────────────────────────────────────────────
KAGGLE_ROOT = PROJECT_ROOT / 'ArtBench-10'
hf_ds   = load_kaggle_artbench10_splits(KAGGLE_ROOT)
test_hf = hf_ds["test"]

# Loader com Normalize — para GAN e Diffusion
transform = T.Compose([
    T.Resize(32, interpolation=T.InterpolationMode.BILINEAR),
    T.CenterCrop(32),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
test_ds     = HFDatasetTorch(test_hf, transform=transform)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=True, num_workers=0)

# Loader sem Normalize — para VAE (treinou com [0,1])
transform_vae = T.Compose([
    T.Resize(32, interpolation=T.InterpolationMode.BILINEAR),
    T.CenterCrop(32),
    T.ToTensor(),
])
test_ds_vae     = HFDatasetTorch(test_hf, transform=transform_vae)
test_loader_vae = DataLoader(test_ds_vae, batch_size=64, shuffle=True, num_workers=0)

# ── Evaluation config ─────────────────────────────────────────────────────────
evaluation_config = {
    'fid_kid_samples':    5000,
    'num_runs':             10,
    'subset_size':         100,
    'num_subsets':          50,
    'feature_batch_size':   32,
    'generation_seed':     123,
}

all_results = {}

# ─────────────────────────────────────────────────────────────────────────────
# GAN
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("GAN")
print("="*60)

ckpt = torch.load('models/artBenchDCGAN_LS_half_lr_ngf128.pt', map_location=device, weights_only=False)
gan_cfg = ckpt['config']
gan_generator = DCGenerator(latent_dim=gan_cfg['latent_dim'], image_channels=gan_cfg['channels'], ngf=128).to(device)
gan_generator.load_state_dict(ckpt['generator'])
gan_generator.eval()

all_results['DCGAN_LS_HalfLR_ngf128_300_epochs'] = run_evaluation(
    generator   = gan_generator,
    latent_dim  = gan_cfg['latent_dim'],
    ref_loader  = test_loader,
    device      = device,
    cfg         = evaluation_config,
    generate_fn = generate_images,
)

# ─────────────────────────────────────────────────────────────────────────────
# Diffusion
# ─────────────────────────────────────────────────────────────────────────────

# for config_name, ckpt_name in [
#     ('baseline',  'baseline_final.pt'),
#     ('cfg_w1',    'cfg_w1_final.pt'),
#     ('cfg_w3',    'cfg_w3_final.pt'),
#     ('cap_64',    'cap_64_final.pt'),
# ]:
#     print(f"\n{'='*60}\nDiffusion {config_name}\n{'='*60}")

#     cfg_i = CONFIGS[config_name]
#     cfg_i.ddim_steps = 20
#     sched_i   = make_cosine_schedule(cfg_i.T, s=cfg_i.cosine_s, device=device)
#     diff_i    = GaussianDiffusion(sched_i, device)

#     ckpt_i    = torch.load(f'models/{ckpt_name}', map_location=device, weights_only=False)
#     model_i   = ArtBenchUNet(
#         model_channels=cfg_i.model_channels,
#         num_classes=cfg_i.num_classes,
#         use_cfg=cfg_i.use_cfg,
#     ).to(device)
#     model_i.load_state_dict(ckpt_i['ema'] if ckpt_i.get('ema') else ckpt_i['model'])
#     model_i.eval()

#     def make_diff_fn(m, d, c):
#         def fn(model, latent_dim, n, device, seed):
#             return generate_samples(m, d, c, device=device, n=n, batch_size=128, use_ddim=True, seed=seed)
#         return fn

#     all_results[f'Diffusion_{config_name}_{cfg_i.epochs}ep'] = run_evaluation(
#         generator   = model_i,
#         latent_dim  = None,
#         ref_loader  = test_loader,
#         device      = device,
#         cfg         = evaluation_config,
#         generate_fn = make_diff_fn(model_i, diff_i, cfg_i),
#     )

# ─────────────────────────────────────────────────────────────────────────────
# VAE
# ─────────────────────────────────────────────────────────────────────────────
# print("\n" + "="*60)
# print("VAE")
# print("="*60)

# vae_model = ConvVAE(latent_dim=128).to(device)
# vae_model.load_state_dict(torch.load('models/artbench_beta_vae_1_300ep.pt', map_location=device))
# vae_model.eval()

# _vae_gen = make_generate_fn(vae_model, device)

# def vae_generate_fn(model, latent_dim, n, device, seed):
#     return _vae_gen(n=n, batch_size=128, seed=seed)

# all_results['Beta_VAE_1_300_epochs'] = run_evaluation(
#     generator   = vae_model,
#     latent_dim  = 128,
#     ref_loader  = test_loader_vae,
#     device      = device,
#     cfg         = evaluation_config,
#     generate_fn = vae_generate_fn,
# )
# ─────────────────────────────────────────────────────────────────────────────
# Real vs Real — sanity check
# ─────────────────────────────────────────────────────────────────────────────
# print("\n" + "="*60)
# print("Real vs Real (sanity check)")
# print("="*60)

# def real_generate_fn(model, latent_dim, n, device, seed):
#     torch.manual_seed(seed)
#     all_real = []
#     for batch in test_loader:
#         all_real.append(batch[0])
#         if sum(x.shape[0] for x in all_real) >= n:
#             break
#     imgs = torch.cat(all_real, dim=0)[:n]
#     idx = torch.randperm(len(imgs))
#     return imgs[idx]

# all_results['Real_vs_Real'] = run_evaluation(
#     generator   = None,
#     latent_dim  = None,
#     ref_loader  = test_loader,
#     device      = device,
#     cfg         = evaluation_config,
#     generate_fn = real_generate_fn,
# )
# ─────────────────────────────────────────────────────────────────────────────
# Sumário comparativo
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"{'MODELO':>15s}  {'FID':>12s}  {'KID':>15s}")
print("="*60)
for model_name, res in all_results.items():
    print(
        f"  {model_name:>13s}  "
        f"{res['fid_mean']:7.4f} ±{res['fid_std']:6.4f}  "
        f"{res['kid_mean']:.6f} ±{res['kid_std']:.6f}"
    )
print("="*60)

# ── Guardar resultados ────────────────────────────────────────────────────────
Path('histories').mkdir(exist_ok=True)
Path('histories/evaluation_results2.json').write_text(
    json.dumps(all_results, indent=2)
)
print("\nResultados guardados em histories/evaluation_results2.json") 