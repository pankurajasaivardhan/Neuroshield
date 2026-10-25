"""Track 2: brain tumor segmentation (BraTS2020, friend's trained 2D nnU-Net).

Uses nnunetv2's official inference API against the recovered trainer folder
(plans.json + dataset.json + fold_0/checkpoint_best.pth). The optional
FLAIR-removal counterfactual re-runs the same case with the FLAIR modality
zeroed, reproducing the project's real robustness experiment
(51/56 held-out test cases went unstable without FLAIR).
"""
from __future__ import annotations

import os
import json
import shutil
import subprocess
import tempfile
import numpy as np
import nibabel as nib

MODALITIES = ["flair", "t1", "t1ce", "t2"]
LABEL_NAMES = {1: "Necrotic / non-enhancing core", 2: "Swelling (edema)", 3: "Enhancing tumor"}
LABEL_COLORS = {1: (59, 130, 246), 2: (34, 197, 94), 3: (239, 68, 68)}  # blue, green, red


class Track2Runner:
    def __init__(self, model_dir: str):
        # model_dir: .../Dataset001_BraTS2020/nnUNetTrainer_100epochs__nnUNetPlans__2d
        self.model_dir = model_dir
        os.environ.setdefault("nnUNet_results", os.path.dirname(os.path.dirname(model_dir)))
        os.environ.setdefault("nnUNet_raw", os.path.join(os.environ["nnUNet_results"], "..", "nnUNet_raw"))
        os.environ.setdefault("nnUNet_preprocessed", os.path.join(os.environ["nnUNet_results"], "..", "nnUNet_preprocessed"))
        self._predictor = None

    def _get_predictor(self):
        if self._predictor is None:
            import torch
            from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
            predictor = nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=False,  # test-time augmentation off for CPU demo speed; disclosed in UI
                perform_everything_on_device=False,
                device=torch.device("cpu"),
                verbose=False,
                verbose_preprocessing=False,
                allow_tqdm=False,
            )
            predictor.initialize_from_trained_model_folder(self.model_dir, use_folds=(0,), checkpoint_name="checkpoint_final.pth")
            self._predictor = predictor
        return self._predictor

    def _predict(self, modality_paths: dict[str, str], out_dir: str, case_id: str) -> str:
        predictor = self._get_predictor()
        files = [[modality_paths[m] for m in MODALITIES]]
        out_file = os.path.join(out_dir, f"{case_id}.nii.gz")
        predictor.predict_from_files(files, [out_file], save_probabilities=False, overwrite=True,
                                     num_processes_preprocessing=1, num_processes_segmentation_export=1)
        return out_file

    @staticmethod
    def _zero_flair(modality_paths: dict[str, str], work_dir: str) -> dict[str, str]:
        img = nib.load(modality_paths["flair"])
        zeroed = nib.Nifti1Image(np.zeros(img.shape, dtype=np.float32), img.affine, img.header)
        out = os.path.join(work_dir, "flair_removed.nii.gz")
        nib.save(zeroed, out)
        return {**modality_paths, "flair": out}

    @staticmethod
    def _dice(a: np.ndarray, b: np.ndarray) -> float:
        a = a.astype(bool); b = b.astype(bool)
        inter = np.logical_and(a, b).sum()
        denom = a.sum() + b.sum()
        return float(2 * inter / denom) if denom else 1.0

    @staticmethod
    def _volumes_cm3(seg: np.ndarray, affine: np.ndarray) -> dict:
        voxel_cm3 = abs(float(np.prod(np.diag(affine)[:3]))) / 1000.0
        out = {}
        for lab, name in LABEL_NAMES.items():
            out[name] = round(float((seg == lab).sum()) * voxel_cm3, 2)
        out["Whole tumor"] = round(float((seg > 0).sum()) * voxel_cm3, 2)
        return out

    def run(self, modality_paths: dict[str, str], case_id: str, work_dir: str,
            remove_flair: bool, gt_path: str | None = None, status_cb=None) -> dict:
        def status(msg):
            if status_cb:
                status_cb(msg)

        status("Running segmentation on the full 4-sequence scan")
        seg_path = self._predict(modality_paths, work_dir, case_id + "_full")
        seg_img = nib.load(seg_path)
        seg = np.asanyarray(seg_img.dataobj).astype(np.uint8)

        result = {
            "case_id": case_id,
            "volumes_cm3": self._volumes_cm3(seg, seg_img.affine),
            "labels": LABEL_NAMES,
            "tumor_found": bool((seg > 0).any()),
        }

        if gt_path and os.path.exists(gt_path):
            status("Comparing with the expert outline")
            gt = np.asanyarray(nib.load(gt_path).dataobj).astype(np.uint8)
            gt = np.where(gt == 4, 3, gt)  # BraTS original labels 1,2,4 -> trained labels 1,2,3
            dice = {"Whole tumor": round(self._dice(seg > 0, gt > 0), 4)}
            for lab, name in LABEL_NAMES.items():
                dice[name] = round(self._dice(seg == lab, gt == lab), 4)
            result["vs_expert_dice"] = dice

        if remove_flair:
            status("Re-running the same scan with the FLAIR sequence removed")
            cf_paths = self._zero_flair(modality_paths, work_dir)
            cf_seg_path = self._predict(cf_paths, work_dir, case_id + "_noflair")
            cf_seg = np.asanyarray(nib.load(cf_seg_path).dataobj).astype(np.uint8)
            wt_dice = self._dice(seg > 0, cf_seg > 0)
            result["flair_removal"] = {
                "whole_tumor_stability_dice": round(wt_dice, 4),
                "stable": wt_dice > 0.8,
                "explanation": (
                    "Stability Dice measures how similar the tumor outline is with and without "
                    "the FLAIR sequence (1.0 = identical). In the project's held-out test, 51 of "
                    "56 cases fell to 0.8 or below without FLAIR, meaning the model leans heavily "
                    "on that sequence."
                ),
            }
            self._make_overlays(modality_paths["t1"], cf_seg, work_dir, case_id + "_noflair")
            result["flair_removal"]["overlay_slices"] = self._overlay_urls(case_id + "_noflair", work_dir)

        self._make_overlays(modality_paths["t1"], seg, work_dir, case_id)
        result["overlay_slices"] = self._overlay_urls(case_id, work_dir)
        return result

    def _make_overlays(self, bg_path: str, seg: np.ndarray, work_dir: str, prefix: str) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        bg = np.asanyarray(nib.load(bg_path).dataobj).astype(np.float32)
        counts = (seg > 0).sum(axis=(0, 1))
        if counts.sum() == 0:
            slices = [seg.shape[2] // 2]
        else:
            slices = list(np.argsort(counts)[-3:])
        out_dir = os.path.join(work_dir, "overlays")
        os.makedirs(out_dir, exist_ok=True)
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap([(0, 0, 0, 0)] + [tuple(c / 255 for c in LABEL_COLORS[i]) + (0.45,) for i in (1, 2, 3)])
        for z in sorted(slices):
            fig, ax = plt.subplots(figsize=(4, 4), dpi=110)
            sl = bg[:, :, z].T[::-1]
            p1, p99 = np.percentile(sl[sl > 0], [1, 99]) if (sl > 0).any() else (0, 1)
            ax.imshow(np.clip((sl - p1) / (p99 - p1 + 1e-8), 0, 1), cmap="gray")
            ax.imshow(seg[:, :, z].T[::-1], cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
            ax.set_title(f"Slice {z}", fontsize=10)
            ax.axis("off")
            fig.tight_layout(pad=0.3)
            fig.savefig(os.path.join(out_dir, f"{prefix}_z{z:03d}.png"))
            plt.close(fig)

    @staticmethod
    def _overlay_urls(prefix: str, work_dir: str) -> list[str]:
        out_dir = os.path.join(work_dir, "overlays")
        if not os.path.isdir(out_dir):
            return []
        return sorted(f"overlays/{f}" for f in os.listdir(out_dir) if f.startswith(prefix))
