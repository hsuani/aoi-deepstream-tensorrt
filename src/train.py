"""
Train YOLOv8-seg on a single MVTec class for AOI defect segmentation.

Usage:
    python src/train.py --class transistor --epochs 50 --batch 8
"""
import argparse
import json
import os
from pathlib import Path
import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8-seg on a single MVTec class for AOI defect segmentation."
    )
    parser.add_argument("--class", dest='class_name', type=str, required=True,
                        choices=['transistor', 'cable', 'metal_nut'],
                        help="MVTec class to train on")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default='mps')
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--name", type=str, default=None,
                        help="Run name; auto = '<class>_v1' if omitted")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--data-root", type=Path, default=Path('data/yolo'))
    parser.add_argument("--resume", action='store_true',
                        help="Resume from runs/segment/<name>/weights/last.pt")

    args = parser.parse_args()

    if args.name is None:
        args.name = f'{args.class_name}_v1'
    return args


def build_train_kwargs(args) -> dict:
    """Compose kwargs dict for YOLO.train(). Apply AOI-tuned augmentation."""
    return {
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "workers": args.workers,
        "name": args.name,
        "seed": args.seed,
        "patience": args.patience,

        "single_cls": True,
        "amp": False,
        "cache": False,

        "plots": True,
        "verbose": True,
        "exist_ok": True,

        "hsv_h": 0.015,
        "hsv_s": 0.5,
        "hsv_v": 0.4,
        "degrees": 10,
        "translate": 0.1,
        "scale": 0.3,
        "shear": 0,
        "perspective": 0,
        "flipud": 0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.1,
        "copy_paste": 0.3,
        "close_mosaic": 10,
        "erasing": 0.4,
    }


def verify_data_yaml(args) -> Path:
    """Check data.yaml exists and content sane. Raise FileNotFoundError / ValueError if not."""
    path = args.data_root / args.class_name / 'data.yaml'
    if not path.exists():
        raise FileNotFoundError(
            f'data.yaml not found at {path}. Run mvtec_to_yolo.py first.'
        )

    with open(path) as f:
        cfg = yaml.safe_load(f)

    required_keys = ('train', 'val', 'names')
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise ValueError(f'{path} missing required keys: {missing}')

    return path


def main():
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    args = parse_args()
    data_yaml = verify_data_yaml(args)

    from ultralytics import YOLO

    weights_path = Path('models/yolov8s-seg.pt')
    weights = weights_path if weights_path.exists() else 'yolov8s-seg.pt'
    model = YOLO(weights)

    train_kwargs = build_train_kwargs(args)
    train_kwargs['data'] = str(data_yaml)

    if args.resume:
        results = model.train(resume=True)
    else:
        results = model.train(**train_kwargs)

    save_dir = Path(results.save_dir) if hasattr(results, 'save_dir') else None

    if save_dir and not args.resume:
        (save_dir / 'aoi_train_kwargs.json').write_text(
            json.dumps(train_kwargs, default=str, indent=2)
        )

    if save_dir:
        print(f'\n[done] best weights: {save_dir / "weights" / "best.pt"}')
    else:
        print('\n[done] (see runs/segment/...)')

    if hasattr(results, 'results_dict'):
        rd = results.results_dict
        for key in (
            'metrics/precision(M)', 'metrics/recall(M)',
            'metrics/mAP50(M)', 'metrics/mAP50-95(M)',
            'metrics/mAP50(B)', 'metrics/mAP50-95(B)',
        ):
            if key in rd:
                print(f'  {key}: {rd[key]:.4f}')


if __name__ == '__main__':
    main()