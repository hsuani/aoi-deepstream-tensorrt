"""Draw YOLO seg labels on image, save overlay PNG."""
import argparse
from pathlib import Path
import cv2
import numpy as np


def parse_yolo_label(label_path: Path) -> list[tuple[int, list[float]]]:
    """Return list of (class_id, [x1,y1,x2,y2,...] normalized)."""
    with open(label_path, 'r') as f:
        result = []
        for line in f:
            if line.strip():
                token = line.split(' ')
                class_id = int(token[0])
                coords = [float(t) for t in token[1:]]
                result.append((class_id, coords))
    return result



def overlay_polygons(image: np.ndarray, items: list, color=(0, 255, 0)) -> np.ndarray:
    """Draw polygons on image (in-place is fine, but return for chaining)."""
    h, w = image.shape[:2]
    out = image.copy() 
    for cls_id, coords in items:
        pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] *= w
        pts[:, 1] *= h
        pts = pts.astype(np.int32)
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], color=color)
        out = cv2.addWeighted(out, 0.7, overlay, 0.3, 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img', type=Path, required=True)
    ap.add_argument('--label', type=Path)
    ap.add_argument('--out', type=Path, default=Path('docs/yolo_overlay.png'))
    args = ap.parse_args()

    if args.label is None:
        # auto-derive: data/yolo/<cls>/images/train/X.png  →  labels/train/X.txt
        args.label = (args.img.parent.parent.parent / 'labels' /
                      args.img.parent.name / f'{args.img.stem}.txt')

    img = cv2.imread(str(args.img))
    items = parse_yolo_label(args.label)
    out = overlay_polygons(img, items)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), out)
    print(f'wrote {args.out}, polygons={len(items)}')


if __name__ == '__main__':
    main()