"""
Convert MVTec AD dataset to Ultralytics YOLOv8-seg format.

Per-class binary segmentation: class 0 = defect.
Train negatives = train/good. Val negatives = test/good.
Train/val positives split from test/<defect_type> by --val-ratio.
"""
import argparse
import random
import shutil
from pathlib import Path
import cv2
import numpy as np
import yaml


def mask_to_polygons(mask_path: Path, eps_ratio: float = 0.002) -> list[list[float]]:
    """
    Read a binary mask, return normalized polygons.

    Returns: list of polygons, each polygon = [x1, y1, x2, y2, ..., xn, yn] normalized to [0,1].
    """
    mask_path = Path(mask_path)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    h, w = mask.shape
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    filtered = []
    for contour in contours:
        arcLength = cv2.arcLength(contour, closed=True)
        epsilon = eps_ratio * arcLength
        simplified = cv2.approxPolyDP(contour, epsilon, closed=True)
        if len(simplified) >= 3:
            flat = simplified.reshape(-1).tolist()
            normalized = [val / w if i % 2 == 0 else val / h for i, val in enumerate(flat)]
            filtered.append(normalized)

    return filtered


def write_yolo_label(label_path: Path, polygons: list[list[float]], class_id: int = 0) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, 'w') as f:
        for poly in polygons:
            coords = ' '.join(f'{c:.6f}' for c in poly)
            f.write(f'{class_id} {coords}\n')


def collect_samples(mvtec_class_dir: Path) -> tuple[list, list, list]:
    train_neg = sorted((mvtec_class_dir / 'train' / 'good').glob('*.png'))
    test_neg = sorted((mvtec_class_dir / 'test' / 'good').glob('*.png'))
    
    defect_samples = []
    test_root = mvtec_class_dir / 'test'
    gt_root = mvtec_class_dir / 'ground_truth'
    for defect_dir in sorted(gt_root.iterdir()):
        if not defect_dir.is_dir():
            continue
        defect_type = defect_dir.name
        img_dir = test_root / defect_type
        for mask_path in sorted(defect_dir.glob('*_mask.png')):
            img_path = img_dir / f'{mask_path.stem.replace("_mask", "")}.png'
            if img_path.exists():
                defect_samples.append((defect_type, img_path, mask_path))
            else:
                print(f'[warn] missing image for {mask_path}')
    
    return train_neg, test_neg, defect_samples

def link_or_copy(src: Path, dst: Path, use_symlink: bool = True) -> None:
    """Create symlink dst -> src, or copy if symlink fails."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    src_abs = src.resolve()
    if use_symlink:
        try:
            dst.symlink_to(src_abs)
            return
        except OSError as e:
            print(f'[warn] symlink failed ({e}); copying {src} → {dst}')
    shutil.copy2(src_abs, dst)


def write_data_yaml(yolo_class_dir: Path, class_name: str) -> None:
    """Write data.yaml for Ultralytics training."""
    cfg = {
        'path': str(yolo_class_dir.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'defect'},
    }
    with open(yolo_class_dir / 'data.yaml', 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def convert_class(
    mvtec_class_dir: Path,
    yolo_class_dir: Path,
    val_ratio: float,
    rng: random.Random,
) -> None:
    """Convert one MVTec class to YOLO seg dataset."""
    train_neg, test_neg, defect_samples = collect_samples(mvtec_class_dir)

    defect_samples = list(defect_samples)
    rng.shuffle(defect_samples)
    n_val = int(len(defect_samples) * val_ratio)
    val_defects = defect_samples[:n_val]
    train_defects = defect_samples[n_val:]

    img_train = yolo_class_dir / 'images' / 'train'
    img_val = yolo_class_dir / 'images' / 'val'
    lbl_train = yolo_class_dir / 'labels' / 'train'
    lbl_val = yolo_class_dir / 'labels' / 'val'
    for d in (img_train, img_val, lbl_train, lbl_val):
        d.mkdir(parents=True, exist_ok=True)

    n_train_pos = n_val_pos = 0

    for src in train_neg:
        name = f'good_train_{src.stem}'
        link_or_copy(src, img_train / f'{name}.png')
        write_yolo_label(lbl_train / f'{name}.txt', polygons=[])

    for src in test_neg:
        name = f'good_test_{src.stem}'
        link_or_copy(src, img_val / f'{name}.png')
        write_yolo_label(lbl_val / f'{name}.txt', polygons=[])

    for defect_type, img, mask in train_defects:
        name = f'{defect_type}_{img.stem}'
        link_or_copy(img, img_train / f'{name}.png')
        polygons = mask_to_polygons(mask)
        write_yolo_label(lbl_train / f'{name}.txt', polygons, class_id=0)
        if polygons:
            n_train_pos += 1

    for defect_type, img, mask in val_defects:
        name = f'{defect_type}_{img.stem}'
        link_or_copy(img, img_val / f'{name}.png')
        polygons = mask_to_polygons(mask)
        write_yolo_label(lbl_val / f'{name}.txt', polygons, class_id=0)
        if polygons:
            n_val_pos += 1

    write_data_yaml(yolo_class_dir, mvtec_class_dir.name)

    print(
        f'  train: {len(train_neg) + len(train_defects)} '
        f'(neg={len(train_neg)}, pos_with_polygon={n_train_pos}/{len(train_defects)})'
    )
    print(
        f'  val:   {len(test_neg) + len(val_defects)} '
        f'(neg={len(test_neg)}, pos_with_polygon={n_val_pos}/{len(val_defects)})'
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', type=Path, default=Path('data/mvtec'))
    ap.add_argument('--dst', type=Path, default=Path('data/yolo'))
    ap.add_argument('--classes', nargs='+', default=['transistor', 'cable', 'metal_nut'])
    ap.add_argument('--val-ratio', type=float, default=0.3)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    for cls in args.classes:
        src_cls = args.src / cls
        dst_cls = args.dst / cls
        if not src_cls.exists():
            print(f'[skip] {src_cls} not found')
            continue
        print(f'[convert] {cls}')
        convert_class(src_cls, dst_cls, args.val_ratio, rng)


if __name__ == '__main__':
    main()