import numpy as np
from scipy import linalg
import torch
from torchvision.models import inception_v3, Inception_V3_Weights


def load_inception(device: torch.device):
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    return model


@torch.no_grad()
def get_inception_features(images: torch.Tensor, model, device, batch_size: int = 32):
    imgs = images.float()
    if imgs.min() < 0:
        imgs = (imgs * 0.5 + 0.5).clamp(0, 1)

    imgs = torch.nn.functional.interpolate(imgs, size=(299, 299), mode='bilinear', align_corners=False)

    feats = []
    for i in range(0, len(imgs), batch_size):
        feats.append(model(imgs[i:i + batch_size].to(device)).cpu().numpy())
    return np.concatenate(feats, axis=0)


# FID 
def compute_fid(real_feats: np.ndarray, fake_feats: np.ndarray):
    mu1, sigma1 = real_feats.mean(0), np.cov(real_feats, rowvar=False)
    mu2, sigma2 = fake_feats.mean(0), np.cov(fake_feats, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


# KID
def compute_kid(real_feats: np.ndarray, fake_feats: np.ndarray, num_subsets: int = 50, subset_size: int = 100):
    subset_size = min(subset_size, len(real_feats), len(fake_feats))
    d = real_feats.shape[1]
    scores = []

    for _ in range(num_subsets):
        r = real_feats[np.random.choice(len(real_feats), subset_size, replace=False)]
        f = fake_feats[np.random.choice(len(fake_feats), subset_size, replace=False)]

        kxx = (r @ r.T / d + 1) ** 3
        kyy = (f @ f.T / d + 1) ** 3
        kxy = (r @ f.T / d + 1) ** 3

        mmd = (kxx.sum() - np.diag(kxx).sum()) / (subset_size * (subset_size - 1)) \
            + (kyy.sum() - np.diag(kyy).sum()) / (subset_size * (subset_size - 1)) \
            - 2 * kxy.mean()
        scores.append(mmd)

    return float(np.mean(scores)), float(np.std(scores))



def run_evaluation(generator, latent_dim: int, ref_loader, device: torch.device, cfg: dict, generate_fn=None):
    n = cfg.get('fid_kid_samples', 5000)
    n_runs = cfg.get('num_runs', 10)
    n_sub = cfg.get('num_subsets', 50)
    sub_size = cfg.get('subset_size', 100)
    bs = cfg.get('feature_batch_size', 32)
    base_seed = cfg.get('generation_seed', 123)

    gen_fn = generate_fn

    print(f"Protocol: {n} generated vs {n} real  |  "
          f"{n_runs} runs  |  KID: {n_sub} subsets × {sub_size}")

    inception = load_inception(device)

    all_real = []
    for batch in ref_loader:
        all_real.append(batch[0])
        if sum(x.shape[0] for x in all_real) >= n:
            break
    all_real = torch.cat(all_real, dim=0)[:n]

    print(f"\nA extrair features das imagens reais (fixas, n={len(all_real)})...")
    feats_real = get_inception_features(all_real, inception, device, bs)

    fid_scores, kid_means, kid_stds = [], [], []

    for run in range(n_runs):
        seed = base_seed + run
        print(f"Run {run + 1}/{n_runs}  (seed={seed})")

        gen_imgs = gen_fn(generator, latent_dim, n, device, seed)

        print("A extrair features das imagens geradas...")
        feats_gen = get_inception_features(gen_imgs, inception, device, bs)

        fid_val = compute_fid(feats_real, feats_gen)
        kid_mean, kid_std = compute_kid(feats_real, feats_gen, n_sub, sub_size)

        fid_scores.append(fid_val)
        kid_means.append(kid_mean)
        kid_stds.append(kid_std)

        print(f"FID = {fid_val:.4f}  |  KID = {kid_mean:.6f} ± {kid_std:.6f}")

    print("RESULTADOS FINAIS  (mean ± std  across runs)")
    print(f"FID : {np.mean(fid_scores):.4f} ± {np.std(fid_scores):.4f}")
    print(f"KID : {np.mean(kid_means):.6f} ± {np.std(kid_means):.6f}")

    return {
        'fid_mean':    float(np.mean(fid_scores)),
        'fid_std':     float(np.std(fid_scores)),
        'kid_mean':    float(np.mean(kid_means)),
        'kid_std':     float(np.std(kid_means)),
        'fid_per_run': fid_scores,
        'kid_per_run': kid_means,
    }

