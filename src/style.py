import torch, sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from diffusion import CONFIGS, make_cosine_schedule, GaussianDiffusion, ArtBenchUNet

device = torch.device('mps')
cfg = CONFIGS['best']
cfg.ddim_steps = 50

schedule  = make_cosine_schedule(cfg.T, s=cfg.cosine_s, device=device)
diffusion = GaussianDiffusion(schedule, device)

ckpt = torch.load('models/best_full_final.pt', map_location=device, weights_only=False)
model = ArtBenchUNet(model_channels=cfg.model_channels, num_classes=cfg.num_classes, use_cfg=cfg.use_cfg).to(device)
model.load_state_dict(ckpt['ema'])
model.eval()

class_names = ['Impressionism', 'Realism', 'Romanticism', 'Expressionism', 
               'Baroque', 'Post Impress.', 'Art Nouveau', 'Surrealism', 
               'Ukiyo-e', 'Renaissance']

torch.manual_seed(5000)
fig, axes = plt.subplots(5, 10, figsize=(20, 12))
fig.patch.set_facecolor('black')

with torch.no_grad():
    for row in range(5):
        for class_id in range(10):
            label = torch.tensor([class_id], device=device)
            shape = (1, 3, cfg.image_size, cfg.image_size)
            x = diffusion.p_sample_loop_ddim(model, shape, ddim_steps=50,
                                              eta=0.0, cfg_scale=5.0, labels=label)
            x = (x.clamp(-1, 1) + 1) * 0.5
            img = x.squeeze(0).permute(1, 2, 0).cpu().numpy()
            axes[row, class_id].imshow(img)
            axes[row, class_id].set_facecolor('black')
            axes[row, class_id].axis('off')
            if row == 0:
                axes[row, class_id].set_title(class_names[class_id], fontsize=20, color='white', pad=3)

plt.tight_layout(pad=0.5)
plt.savefig('./samples/per_class_5samples2.png', dpi=150, bbox_inches='tight', facecolor='black')
plt.close()