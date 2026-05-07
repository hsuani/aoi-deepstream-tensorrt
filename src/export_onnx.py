import argparse
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from ultralytics import YOLO


def parse_args():
    """CLI parser."""
    parser = argparse.ArgumentParser(description="Export Ultralytics YOLOv8-seg .pt to ONNX.")
    parser.add_argument("--weights",type=Path, required=True, help="input .pt")
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--simplify', action='store_true')
    parser.add_argument("--half", action="store_true", help="Export FP16 ONNX. Usually skip; let TRT do FP16 at engine build.")
    parser.add_argument("--out", type=Path, help="Optional output .onnx path; default beside .pt")
    return parser.parse_args()


def export(args):
    """Run Ultralytics export. Returns path to .onnx."""
    model = YOLO(args.weights)
    produced = Path(model.export(
        format='onnx',
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=args.dynamic,
        simplify=args.simplify,
        half=args.half,
    ))
    if args.out and produced.resolve() != args.out.resolve():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.out.exists():
            args.out.unlink()
        shutil.move(str(produced), str(args.out))
        return args.out
    return produced


def verify_onnx(onnx_path: Path, imgsz: int) -> None:
    """Load with onnx + onnxruntime, run dummy forward, check output shapes."""
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    print(f"[ok] onnx.checker passed, opset={model.opset_import[0].version}")

    def _shape(node):
        return [d.dim_value or d.dim_param or "?" for d in node.type.tensor_type.shape.dim]

    for inp in model.graph.input:
        print(f"  input  {inp.name}: {_shape(inp)}")
    for out in model.graph.output:
        print(f"  output {out.name}: {_shape(out)}")

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: x})
    print(f"[ok] dummy forward")
    for i, o in enumerate(outputs):
        print(f"  output[{i}].shape: {o.shape}")


def main():
    args = parse_args()
    onnx_path = export(args)
    print(f'[exported] {onnx_path}')
    verify_onnx(onnx_path, args.imgsz)


if __name__ == '__main__':
    main()