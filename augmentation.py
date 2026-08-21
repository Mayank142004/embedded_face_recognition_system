"""
augmentation.py — Lighting-robust face augmentation pipeline.

Usage (standalone):
    python augmentation.py --dataset facenet_files/dataset2 --n_variants 20

Usage (from code):
    from augmentation import augment_dataset
    augment_dataset("facenet_files/dataset2", n_variants=20, status_callback=print)
"""

import os
import cv2
import argparse
import numpy as np

try:
    import albumentations as A
    _ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    _ALBUMENTATIONS_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Build the augmentation pipeline
# ─────────────────────────────────────────────────────────────────────────────
def _build_pipeline() -> "A.Compose":
    """
    Returns a stochastic augmentation pipeline covering:
    - Brightness/contrast/gamma variations  (uneven ambient lighting)
    - CLAHE                                  (local-contrast normalisation)
    - Hue/saturation/value                  (colour-temperature shifts)
    - ISO noise                              (low-light sensor noise)
    - Random shadow                          (directional / side lighting)
    - Small-magnitude ShiftScaleRotate       (slight pose variance; kept
                                              small because FaceNet is
                                              alignment-sensitive)
    """
    if not _ALBUMENTATIONS_AVAILABLE:
        raise ImportError(
            "albumentations is required for augmentation. "
            "Install it with: pip install albumentations"
        )
    return A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=0.35,
            contrast_limit=0.35,
            p=0.8,
        ),
        A.RandomGamma(gamma_limit=(60, 140), p=0.5),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.4),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=30,
            val_shift_limit=30,
            p=0.5,
        ),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.4),
        A.RandomShadow(
            shadow_roi=(0, 0.0, 1, 1),
            num_shadows_lower=1,
            num_shadows_upper=2,
            shadow_dimension=4,
            p=0.4,
        ),
        # Small pose variance — keep limits tight for FaceNet alignment
        A.ShiftScaleRotate(
            shift_limit=0.04,
            scale_limit=0.06,
            rotate_limit=8,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5,
        ),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Per-image augmentation
# ─────────────────────────────────────────────────────────────────────────────
def generate_augmented_faces(
    image_path: str,
    output_dir: str,
    n_variants: int = 20,
    pipeline: "A.Compose | None" = None,
) -> list[str]:
    """
    Read one source image and write ``n_variants`` augmented copies to
    ``output_dir``.

    Args:
        image_path:  Path to the source face image (BGR, any size).
        output_dir:  Directory where augmented images will be saved.
        n_variants:  Number of augmented variants to generate.
        pipeline:    Pre-built albumentations Compose pipeline.  If None,
                     one is built on the fly (slightly slower for bulk use).

    Returns:
        List of absolute paths to the written augmented images.
    """
    if pipeline is None:
        pipeline = _build_pipeline()

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    os.makedirs(output_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(image_path))[0]
    written: list[str] = []

    for i in range(n_variants):
        augmented = pipeline(image=img_rgb)["image"]
        aug_bgr   = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
        out_name  = f"{stem}_aug_{i:03d}.jpg"
        out_path  = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, aug_bgr)
        written.append(os.path.abspath(out_path))

    return written


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-level augmentation
# ─────────────────────────────────────────────────────────────────────────────
def augment_dataset(
    dataset_dir: str = "facenet_files/dataset2",
    n_variants: int = 20,
    status_callback=None,
) -> dict:
    """
    Iterate every employee subfolder in ``dataset_dir``, augment each real
    source image, and write the results back into the same folder.

    Images already named ``*_aug_*.jpg`` are skipped so that re-running this
    function on an already-augmented dataset does not compound variants.

    Args:
        dataset_dir:      Root dataset directory (one sub-folder per person).
        n_variants:       Number of augmented images to produce per source image.
        status_callback:  Optional callable(str) for progress reporting —
                          accepts the same interface as training.py's
                          status_callback so it slots straight into the
                          dashboard's append_log() panel.

    Returns:
        dict with 'total_source' (count of real images processed) and
        'total_generated' (count of augmented images written).
    """
    def log(msg: str) -> None:
        print(msg)
        if status_callback:
            status_callback(msg)

    if not _ALBUMENTATIONS_AVAILABLE:
        log("⚠️  albumentations not installed — skipping augmentation.")
        log("   Install with: pip install albumentations")
        return {"total_source": 0, "total_generated": 0}

    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    pipeline = _build_pipeline()
    total_source = 0
    total_generated = 0

    employees = sorted(os.listdir(dataset_dir))
    log(f"Augmenting dataset in: {dataset_dir}  ({len(employees)} employee folders)")

    for emp_name in employees:
        emp_dir = os.path.join(dataset_dir, emp_name)
        if not os.path.isdir(emp_dir):
            continue

        # Collect real source images — skip previously generated aug files
        source_images = [
            f for f in os.listdir(emp_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            and '_aug_' not in f
        ]

        if not source_images:
            log(f"  [{emp_name}] No source images found — skipping.")
            continue

        log(f"  [{emp_name}] {len(source_images)} source image(s) → "
            f"{len(source_images) * n_variants} augmented variants …")

        emp_generated = 0
        for fname in source_images:
            fpath = os.path.join(emp_dir, fname)
            try:
                written = generate_augmented_faces(
                    image_path=fpath,
                    output_dir=emp_dir,
                    n_variants=n_variants,
                    pipeline=pipeline,
                )
                emp_generated += len(written)
                total_source  += 1
            except Exception as exc:
                log(f"    ⚠️  Skipped {fname}: {exc}")

        total_generated += emp_generated
        log(f"  [{emp_name}] ✅ {emp_generated} images written.")

    log(f"Augmentation complete. "
        f"{total_source} source images → {total_generated} augmented images total.")

    return {"total_source": total_source, "total_generated": total_generated}


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate augmented face images for the FaceAttend dataset."
    )
    parser.add_argument(
        "--dataset",
        default="facenet_files/dataset2",
        help="Root dataset directory (one sub-folder per employee).",
    )
    parser.add_argument(
        "--n_variants",
        type=int,
        default=20,
        help="Number of augmented variants to generate per source image.",
    )
    args = parser.parse_args()

    result = augment_dataset(
        dataset_dir=args.dataset,
        n_variants=args.n_variants,
        status_callback=print,
    )
    print(f"\nSummary: {result['total_source']} source → "
          f"{result['total_generated']} generated images.")
