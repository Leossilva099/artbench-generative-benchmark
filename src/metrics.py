import torch
import numpy as np
from torchvision import transforms as T

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
        generate_fn,
        hf_split, image_size, device,
        n_samples=5000, kid_subsets=50, kid_subset_size=100,
        gen_batch_size=128, seed=0,
    ) -> dict:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance

    # Em vez de chamar a difusão, chama a função genérica que lhe passaste
    fake = generate_fn(n=n_samples, batch_size=gen_batch_size, seed=seed)
    
    fake_u8 = (fake * 255).to(torch.uint8)
    real_u8 = get_real_images_uint8(hf_split, n_samples, seed, image_size)

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
    generate_fn,
    hf_split, image_size, device,
    n_seeds=10, n_samples=5000, gen_batch_size=128,
) -> dict:
    fids, k_means, k_stds = [], [], []

    for seed in range(n_seeds):
        r = compute_fid_kid(
            generate_fn, hf_split, image_size, device,
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
