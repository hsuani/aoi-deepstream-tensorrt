# DeepStream 9.0 runtime + dev tools for YOLOv8-seg-det metal-nut pipeline.
# Build context = src/deepstream/models/yolov8s_seg_metal_nut/
#
# Build:
#   docker build -t aoi-metal-nut:ds9 -f docker/Dockerfile .
# Run (engine build + multi-stream bench):
#   docker run --rm --gpus all \
#     -v $PWD/model:/app/model \
#     -v $PWD/benchmarks:/app/benchmarks \
#     -v $PWD/samples:/app/samples \
#     -v $PWD/sources:/app/sources \
#     -v /path/to/mvtec/metal_nut/test:/data/mvtec/metal_nut/test:ro \
#     aoi-metal-nut:ds9 all

FROM nvcr.io/nvidia/deepstream:9.0-triton-multiarch

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    GST_DEBUG=1 \
    CUDA_VER=12.8 \
    DEEPSTREAM_DIR=/opt/nvidia/deepstream/deepstream

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        python3 python3-pip python3-venv \
        git wget curl jq bc \
        libgstreamer1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad gstreamer1.0-libav gstreamer1.0-tools \
        wkhtmltopdf \
    && rm -rf /var/lib/apt/lists/*

# pyservicemaker wheel ships in DS 9.0 sources; install into system Python.
RUN PSM_WHL=$(ls $DEEPSTREAM_DIR/service-maker/sources/python/wheels/pyservicemaker-*.whl 2>/dev/null \
               || ls $DEEPSTREAM_DIR/sources/python/pyservicemaker/wheels/pyservicemaker-*.whl 2>/dev/null \
               | head -1) && \
    if [ -n "$PSM_WHL" ]; then pip3 install --no-cache-dir "$PSM_WHL"; \
    else echo "WARN: pyservicemaker wheel not found in image"; fi

RUN pip3 install --no-cache-dir \
        numpy matplotlib markdown Pillow onnxruntime

WORKDIR /app
COPY model/      /app/model/
COPY parser/     /app/parser/
COPY config/     /app/config/
COPY benchmarks/ /app/benchmarks/
COPY scripts/    /app/scripts/
COPY samples/    /app/samples/
COPY sources/    /app/sources/
COPY reports/    /app/reports/

# Build parser library at image build time so a missing toolchain is caught early.
RUN make -C /app/parser \
        DEEPSTREAM_DIR=$DEEPSTREAM_DIR \
        CUDA_VER=$CUDA_VER && \
    nm -D /app/parser/libnvdsinfer_yolov8s_seg_metal_nut_parser.so | grep NvDsInferParseCustomYolov8SegMetalNut

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh \
    && chmod +x /app/scripts/*.sh /app/scripts/*.py

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["all"]
