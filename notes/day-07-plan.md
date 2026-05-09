# Day 7 — Plan: Run pipeline, generate PDF, demo, optional seg extension

## Sequence

### 7a. Stage model artifacts into skill model dir (Mac, 5 min)
```bash
cd /Users/yht/Study/projects/nv-ecosystem/aoi-deepstream-tensorrt
mkdir -p src/deepstream/models/yolov8s_seg_metal_nut/model
cp models/yolov8s_seg_metal_nut.onnx \
   src/deepstream/models/yolov8s_seg_metal_nut/model/
cp engines/metal_nut_int8.cache \
   src/deepstream/models/yolov8s_seg_metal_nut/model/
ls -lh src/deepstream/models/yolov8s_seg_metal_nut/model/
```

### 7b. SCP to GCP VM (Mac, 10 min)
```bash
gcloud compute scp --recurse --zone us-central1-a \
  src/deepstream/models/yolov8s_seg_metal_nut \
  aoi-d5:~/ds-metal-nut

# Also need MVTec metal_nut/test dataset on VM
gcloud compute scp --recurse --zone us-central1-a \
  data/mvtec/metal_nut/test \
  aoi-d5:~/mvtec/metal_nut/test
```

### 7c. Build container + run full pipeline on VM (45-90 min)
```bash
gcloud compute ssh aoi-d5 --zone=us-central1-a

# In VM
cd ~/ds-metal-nut
docker build -t aoi-metal-nut:ds9 -f docker/Dockerfile .

docker run --rm --gpus all \
  -v "$PWD/model":/app/model \
  -v "$PWD/benchmarks":/app/benchmarks \
  -v "$PWD/samples":/app/samples \
  -v "$PWD/sources":/app/sources \
  -v "$PWD/reports":/app/reports \
  -v ~/mvtec/metal_nut/test:/data/mvtec/metal_nut/test:ro \
  aoi-metal-nut:ds9 all
```

`all` mode does:
1. `build_engine.sh` — rebuild ONNX → INT8 engine (BS=1 + BS=4) inside container
2. `prep_test_loops.sh` — generate test_loop_{0..3}.{mp4,ogv}
3. `bench_multistream.sh` — 2 runs × 4 streams, parse **PERF lines
4. `generate_report.py` — 5 charts + md + html + PDF

### 7d. Pull artifacts back (Mac, 5 min)
```bash
gcloud compute scp --recurse --zone us-central1-a \
  aoi-d5:~/ds-metal-nut/reports \
  src/deepstream/models/yolov8s_seg_metal_nut/

gcloud compute scp --recurse --zone us-central1-a \
  "aoi-d5:~/ds-metal-nut/benchmarks/b1/*.log" \
  "aoi-d5:~/ds-metal-nut/benchmarks/b4/*.log" \
  "aoi-d5:~/ds-metal-nut/benchmarks/ds/*.log" \
  src/deepstream/models/yolov8s_seg_metal_nut/benchmarks/

gcloud compute scp --zone us-central1-a \
  aoi-d5:~/ds-metal-nut/samples/metal_nut_mock.mp4 \
  docs/deepstream_demo.mp4
```

### 7e. Commit + push (15 min)
```bash
git add src/deepstream/models/yolov8s_seg_metal_nut/reports \
        src/deepstream/models/yolov8s_seg_metal_nut/benchmarks \
        docs/deepstream_demo.mp4 \
        notes/day-06.md notes/day-07-plan.md notes/day-07.md \
        .gitignore
git commit -m "Day 7: skill-orchestrated DeepStream pipeline run + multi-stream PDF benchmark"
git push
```

### 7f. README v3 update (30 min)
- Tick D5-D7 in roadmap
- Add Day 7 section: multi-stream bench numbers (4 streams aggregate FPS,
  per-stream latency)
- Embed PDF link + key chart PNG
- Update Architecture diagram if needed (mention skill-orchestrated)

### 7g. (Stretch — Path B) Custom seg parser (3-4 hr)

Only if D7 morning bench finishes by 14:00.

Add YOLOv8-seg instance mask parser:

1. New parser file `parser/nvdsinfer_custom_seg_yolov8s_metal_nut.cpp`:
   - Bind `NvDsInferParseCustomInstanceMaskFunc` signature
   - Read output0 [1, 37, 8400] AND output1 [1, 32, 160, 160]
   - For each surviving det (post-NMS):
     - Slice mask coefs ch 5..36 (32 floats)
     - Dot-product against output1 prototype (32 channels) → 1×160×160 raw mask
     - Sigmoid activation
     - Upsample to 640×640
     - Crop to bbox bounds
     - Pack into `NvDsInferInstanceMaskInfo`
2. New nvinfer config `config/config_infer_seg_yolov8s_seg_metal_nut.txt`:
   - `network-type=3` (instance segmentation)
   - `parse-bbox-instance-mask-func-name=NvDsInferParseCustomYOLOv8Seg`
3. Validate single image: `nvdsosd` should overlay polygon mask on top of bbox.
4. README write up: "Extended skill output with custom YOLOv8-seg parser
   recovering instance mask metadata for QA-time defect shape analysis."
5. Demo video v2 with mask overlay.

**Skip if running short on time.** Det-only deployment story is complete
without it.

## Success criteria
- [ ] `reports/benchmark_report_yolov8s_seg_metal_nut.pdf` exists in repo
- [ ] Multi-stream FPS aggregate ≥ 200 fps (4 streams × ~50 fps each on L4 INT8)
- [ ] `docs/deepstream_demo.mp4` ≤ 10 MB, 30 sec, shows bbox + class overlay
  on metal_nut test images
- [ ] README ticks D5 / D6 / D7 in roadmap
- [ ] CV v16 drafted with deployment metrics

## Risks
- Container build fails inside VM → debug Dockerfile; usually missing deps.
- Engine rebuild OOM → reduce trtexec workspace from 4 GB to 2 GB.
- NVENC unavailable on L4 → falls back to theoraenc → `.ogv`. Convert to mp4 with ffmpeg.
- TRT calibration cache version mismatch (Kaggle T4 cache vs GCP L4) →
  cache may rebuild from scratch (~5 min, not blocker).

## Cost projection
- VM uptime D7: ~6 hr × $0.28/hr Spot = $1.70 (cumulative ~$6 / $300 trial)
