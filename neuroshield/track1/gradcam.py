# Grad-CAM saliency for the ResNet reference model.
from __future__ import annotations
import argparse
import numpy as np
import torch

def gradcam(model, target_layer, img: torch.Tensor, class_idx: int | None=None):
    model.eval()
    (activations, gradients) = ([], [])

    def fwd_hook(_m, _i, out):
        activations.append(out.detach())

    def bwd_hook(_m, _gin, gout):
        gradients.append(gout[0].detach())
    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)
    logits = model(img)
    if class_idx is None:
        class_idx = int(logits.argmax(dim=-1))
    model.zero_grad()
    logits[0, class_idx].backward()
    h1.remove()
    h2.remove()
    a = activations[0][0]
    g = gradients[0][0]
    weights = g.mean(dim=(1, 2), keepdim=True)
    cam = torch.relu((weights * a).sum(dim=0)).cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-08)
    return (cam, class_idx)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--model', choices=['a', 'b'], default='b')
    ap.add_argument('--scan-path', required=True)
    ap.add_argument('--out', default='cam.png')
    args = ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import nibabel as nib
    from .data import normalize_mri_slice, IMG_SIZE
    from . import train_model_a, train_model_b
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.model == 'a':
        model = train_model_a.build_model()
        target = model.layer4[-1]
    else:
        model = train_model_b.build_model()
        raise SystemExit('Grad-CAM on ViT needs an attention-rollout variant; use --model a (ResNet) for standard Grad-CAM.')
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    vol = nib.load(args.scan_path).get_fdata().squeeze()
    sl = normalize_mri_slice(vol[vol.shape[0] // 2].astype(np.float32))
    t = torch.from_numpy(sl)[None, None]
    t = torch.nn.functional.interpolate(t, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)[0, 0]
    img = t.repeat(3, 1, 1).unsqueeze(0).to(device)
    (cam, cls) = gradcam(model, target, img)
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(t.numpy(), cmap='gray')
    plt.title('Input slice')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(t.numpy(), cmap='gray')
    plt.imshow(cam, cmap='jet', alpha=0.4)
    plt.title(f'Grad-CAM (class {cls})')
    plt.axis('off')
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print('Saved', args.out)
if __name__ == '__main__':
    main()
