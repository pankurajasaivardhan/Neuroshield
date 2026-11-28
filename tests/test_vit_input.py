import numpy as np, pandas as pd, torch, nibabel as nib
from track1_alzheimers.src import data as D

def _csv(tmp_path, vol):
    img = nib.Nifti1Image(vol[..., None].astype(np.float32), np.eye(4))
    p = tmp_path / 's.nii.gz'; nib.save(img, str(p))
    c = tmp_path / 's.csv'; pd.DataFrame({'scan_path': [str(p)], 'label': [1], 'ID': ['S1']}).to_csv(c, index=False)
    return str(c)

def test_notebook_pipeline_matches_manual(tmp_path):
    rng = np.random.default_rng(0)
    vol = rng.random((256, 256, 128)).astype(np.float32) + 0.1
    ds = D.OASISDatasetNotebookViT(_csv(tmp_path, vol))
    assert len(ds) == 5
    x, y, sid = ds[2]  # offset 0
    c = 128 // 2
    st = np.transpose(vol[:, :, c - 1:c + 2], (2, 0, 1))
    st = D.normalize_mri_slice(st)
    t = torch.nn.functional.interpolate(torch.from_numpy(st)[None], size=(224, 224), mode='bilinear', align_corners=False)[0]
    m = torch.tensor(D.IMAGENET_MEAN).view(3, 1, 1); s = torch.tensor(D.IMAGENET_STD).view(3, 1, 1)
    assert torch.allclose(x, (t - m) / s, atol=1e-5) and y.item() == 1 and sid == 'S1'
    assert x.shape == (3, 224, 224)

def test_offsets_clamped_and_differs_from_default(tmp_path):
    rng = np.random.default_rng(1)
    vol = rng.random((256, 256, 128)).astype(np.float32) + 0.1
    csv = _csv(tmp_path, vol)
    a = D.OASISDatasetNotebookViT(csv); b = D.OASISDataset(csv)
    for i in range(5):
        assert a[i][0].shape == b[i][0].shape == (3, 224, 224)
    assert not torch.allclose(a[2][0], b[2][0])  # the two pipelines are different inputs
    # the default pipeline repeats one slice over the 3 channels, the notebook one does not
    assert torch.allclose(b[2][0][0], b[2][0][1]) and not torch.allclose(a[2][0][0], a[2][0][1])
