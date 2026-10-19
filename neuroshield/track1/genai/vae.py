# VAE prototype + latent counterfactual stability test (GPU).
from __future__ import annotations
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class SliceVAE(nn.Module):

    def __init__(self, latent_dim: int=128):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(), nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(), nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU())
        self.fc_mu = nn.Linear(256 * 14 * 14, latent_dim)
        self.fc_lv = nn.Linear(256 * 14 * 14, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 256 * 14 * 14)
        self.dec = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(), nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(), nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(), nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid())

    def encode(self, x):
        h = self.enc(x).flatten(1)
        return (self.fc_mu(h), self.fc_lv(h))

    def reparam(self, mu, lv):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * lv)

    def decode(self, z):
        return self.dec(self.fc_dec(z).view(-1, 256, 14, 14))

    def forward(self, x):
        (mu, lv) = self.encode(x)
        return (self.decode(self.reparam(mu, lv)), mu, lv)

def vae_loss(recon, x, mu, lv, beta: float=1.0):
    rec = nn.functional.mse_loss(recon, x, reduction='mean')
    kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
    return (rec + beta * kl, rec.item(), kl.item())

def slice_tensor_dataset(data_root: str) -> torch.Tensor:
    import pandas as pd
    import nibabel as nib
    from ..data import normalize_mri_slice
    df = pd.read_csv(os.path.join(data_root, 'splits/train.csv'))
    imgs = []
    for (_, r) in df.iterrows():
        vol = nib.load(r['scan_path']).get_fdata().squeeze()
        sl = normalize_mri_slice(vol[vol.shape[0] // 2].astype(np.float32))
        t = torch.from_numpy(sl)[None, None]
        t = nn.functional.interpolate(t, size=(224, 224), mode='bilinear', align_corners=False)
        imgs.append(t)
    return torch.cat(imgs)

@torch.no_grad()
def latent_perturbation_stability(vae, classifier, slice_batch, sigma: float=0.1, n_samples: int=10, device: str='cpu'):
    vae.eval()
    classifier.eval()
    x = slice_batch.to(device)
    if x.dim() == 3:
        x = x.unsqueeze(1)
    x1 = x if x.shape[1] == 1 else x.mean(dim=1, keepdim=True)
    base = classifier(x1.repeat(1, 3, 1, 1))
    base_pred = base.argmax(dim=-1)
    base_prob = torch.softmax(base, dim=-1)
    (flips, shifts) = (0, [])
    (mu, lv) = vae.encode(x1)
    for _ in range(n_samples):
        z = mu + torch.randn_like(mu) * sigma
        xr = vae.decode(z)
        logits = classifier(xr.repeat(1, 3, 1, 1))
        probs = torch.softmax(logits, dim=-1)
        flips += (logits.argmax(dim=-1) != base_pred).sum().item()
        shifts.append((probs - base_prob).abs().mean().item())
    n = len(x) * n_samples
    return {'flip_rate': flips / n, 'mean_abs_prob_shift': float(np.mean(shifts)), 'sigma': sigma, 'n_samples': n_samples}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--latent-dim', type=int, default=128)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = slice_tensor_dataset(args.data_root)
    loader = DataLoader(data, batch_size=16, shuffle=True)
    vae = SliceVAE(args.latent_dim).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=0.001)
    for epoch in range(args.epochs):
        tot = 0.0
        for x in loader:
            x = x.to(device)
            opt.zero_grad()
            (recon, mu, lv) = vae(x)
            (loss, rec, kl) = vae_loss(recon, x, mu, lv)
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
        print(f'epoch {epoch + 1}: loss={tot / len(data):.4f} rec={rec:.4f} kl={kl:.4f}', flush=True)
    torch.save(vae.state_dict(), os.path.join(args.out, 'vae_slices.pt'))
    print('VAE saved. Next: run latent_perturbation_stability with the trained classifier to test counterfactual stability. Latent diffusion is justified only if this stage works.')
if __name__ == '__main__':
    main()
