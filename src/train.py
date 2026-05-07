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

RECIPES = {
    'v1': {
        'weights': 'yolov8s-seg.pt',
        'imgsz': 640,
        'lr0': 0.01,
        'batch': 8,
        'mixup': 0.1,
        'copy_paste': 0.3,
        'close_mosaic': 10,
        'patience': 25,
    },
    'v2': {
        'weights': 'yolov8n-seg.pt',
        'imgsz': 1024,
        'lr0': 0.001,
        'batch': 4,
        'mixup': 0.0,
        'copy_paste': 0.0,
        'close_mosaic': 20,
        'patience': 15,
    },
    'v3': {
        'weights': 'yolov8s-seg.pt',     # 回 v1 (s 不是 n)
        'imgsz': 640,                    # 回 v1
        'lr0': 0.001,                    # 唯一改動
        'batch': 8,                      # 回 v1
        'mixup': 0.1,                    # 回 v1
        'copy_paste': 0.3,               # 回 v1 ⭐
        'close_mosaic': 20,              # 留 (從 10 → 20，給 model 更多 clean 收尾)
        'patience': 20,
    }
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8-seg on a single MVTec class for AOI defect segmentation."
    )
    parser.add_argument("--class", dest='class_name', type=str, required=True,
                        choices=['transistor', 'cable', 'metal_nut'],
                        help="MVTec class to train on")
    parser.add_argument('--recipe', choices=list(RECIPES.keys()), default='v1')
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
    recipe = RECIPES[args.recipe]

    # base augmentation (recipe-independent)
    base = {
        'data': None,
        'epochs': args.epochs,
        'imgsz': recipe['imgsz'],
        'batch': recipe['batch'],
        'device': args.device,
        'workers': args.workers,
        'name': args.name or f'{args.class_name}_{args.recipe}',
        'seed': args.seed,
        'patience': recipe['patience'],
        'lr0': recipe['lr0'],
        'single_cls': True,
        'amp': False,
        'cache': False,
        'plots': True,
        'verbose': True,
        'exist_ok': True,
        # base augmentation
        'hsv_h': 0.015, 'hsv_s': 0.5, 'hsv_v': 0.4,
        'degrees': 10, 'translate': 0.1, 'scale': 0.3,
        'shear': 0, 'perspective': 0,
        'flipud': 0, 'fliplr': 0.5,
        'erasing': 0.4,
        # recipe-driven augmentation
        'mosaic': 1.0,
        'mixup': recipe['mixup'],
        'copy_paste': recipe['copy_paste'],
        'close_mosaic': recipe['close_mosaic'],
    }
    return base

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

    weights_name = RECIPES[args.recipe]['weights']
    weights_path = Path('models') / weights_name
    weights = str(weights_path) if weights_path.exists() else weights_name

    model = YOLO(weights)
    train_kwargs = build_train_kwargs(args)
    train_kwargs['data'] = str(data_yaml)

    results = model.train(resume=True) if args.resume else model.train(**train_kwargs)

    save_dir = Path(results.save_dir) if hasattr(results, 'save_dir') else None
    print(f'\n[done] best weights: {save_dir / "weights" / "best.pt" if save_dir else "(see runs/segment/...)"}')

if __name__ == '__main__':
    main()