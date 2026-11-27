"""SYNTHETIC check that the all-voxel HD95 used for the reported UNETR numbers is lower than
the standard surface HD95 on the same masks. Does not recompute any reported number."""
import numpy as np
from track2_brain_tumor.src.eval_model_a_test import hd95, hd95_all_voxels


def _ball(shape, c, r):
    z, y, x = np.indices(shape)
    return (z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2 <= r * r


def test_identical_masks_zero():
    a = _ball((40, 40, 40), (20, 20, 20), 8)
    assert hd95(a, a) == 0.0 and hd95_all_voxels(a, a) == 0.0


def test_shifted_ball_surface_hd95_exceeds_all_voxel():
    a = _ball((50, 50, 50), (25, 25, 25), 10)
    b = _ball((50, 50, 50), (25, 25, 31), 10)
    assert hd95(a, b) > hd95_all_voxels(a, b) > 0


if __name__ == '__main__':
    for k, v in list(globals().items()):
        if k.startswith('test_'):
            v(); print('ok', k)
