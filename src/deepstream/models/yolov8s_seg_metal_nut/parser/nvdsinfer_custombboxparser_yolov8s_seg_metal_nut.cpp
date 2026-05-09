// YOLOv8-seg DET-ONLY custom bbox parser for DeepStream 9.0.
//
// Model: yolov8s_seg_metal_nut (single class "defect").
// Outputs (read from layerInfo by name; fallback to index):
//   output0  shape [1, 37, 8400]   = [batch, 4 box + 1 cls + 32 mask_coef, anchors]
//   output1  shape [1, 32, 160, 160] = prototype masks (IGNORED in det-only mode)
//
// Layout (output0): channel-major, contiguous.
//   ptr[c * 8400 + i] for channel c in [0..36], anchor i in [0..8399]
//   c=0..3   : cx, cy, w, h  (pixel coords in network input space, e.g. 640x640)
//   c=4      : class score (sigmoid-applied by ultralytics ONNX export)
//   c=5..36  : 32 mask coefficients (UNUSED here)
//
// NMS: cluster-mode=2 (DeepStream NMS) in nvinfer config.
// pre-cluster-threshold filtering uses detectionParams.perClassPreclusterThreshold.

#include <algorithm>
#include <cassert>
#include <cstring>
#include <cmath>
#include <vector>
#include "nvdsinfer_custom_impl.h"

#define NUM_BBOX_CHANNELS 4
#define NUM_CLASSES_OUT0  1
#define NUM_MASK_COEFS    32
#define EXPECTED_C        (NUM_BBOX_CHANNELS + NUM_CLASSES_OUT0 + NUM_MASK_COEFS)  // 37
#define EXPECTED_ANCHORS  8400

extern "C"
bool NvDsInferParseCustomYolov8SegMetalNut(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList);

static const NvDsInferLayerInfo *findLayer(
    const std::vector<NvDsInferLayerInfo> &layers, const char *name)
{
    for (auto &l : layers) {
        if (l.layerName && std::strcmp(l.layerName, name) == 0) return &l;
    }
    return nullptr;
}

extern "C"
bool NvDsInferParseCustomYolov8SegMetalNut(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
    if (outputLayersInfo.empty()) return false;

    const NvDsInferLayerInfo *out0 = findLayer(outputLayersInfo, "output0");
    if (!out0) out0 = &outputLayersInfo[0];
    if (!out0 || !out0->buffer) return false;

    const float *data = static_cast<const float *>(out0->buffer);

    int channels = 0, anchors = 0;
    if (out0->inferDims.numDims >= 2) {
        channels = out0->inferDims.d[0];
        anchors  = out0->inferDims.d[1];
    }
    if (channels != EXPECTED_C || anchors != EXPECTED_ANCHORS) {
        // Fallback to compile-time constants if dims are missing/dynamic
        channels = EXPECTED_C;
        anchors  = EXPECTED_ANCHORS;
    }

    float threshold = 0.25f;
    if (!detectionParams.perClassPreclusterThreshold.empty()) {
        threshold = detectionParams.perClassPreclusterThreshold[0];
    }

    const float netW = static_cast<float>(networkInfo.width);
    const float netH = static_cast<float>(networkInfo.height);

    objectList.reserve(256);

    for (int i = 0; i < anchors; ++i) {
        float score = data[4 * anchors + i];
        if (score < threshold) continue;

        float cx = data[0 * anchors + i];
        float cy = data[1 * anchors + i];
        float w  = data[2 * anchors + i];
        float h  = data[3 * anchors + i];

        float left = cx - 0.5f * w;
        float top  = cy - 0.5f * h;

        // Clip to network input space; nvinfer rescales to source frame.
        if (left < 0.f) { w += left; left = 0.f; }
        if (top  < 0.f) { h += top;  top  = 0.f; }
        if (left + w > netW - 1.f) w = netW - 1.f - left;
        if (top  + h > netH - 1.f) h = netH - 1.f - top;
        if (w <= 0.f || h <= 0.f) continue;

        // CRITICAL: zero-init for DS 9.0 OBB safety (rotation_angle, etc.).
        NvDsInferObjectDetectionInfo obj = {};
        obj.classId = 0;
        obj.left    = left;
        obj.top     = top;
        obj.width   = w;
        obj.height  = h;
        obj.detectionConfidence = score;

        objectList.push_back(obj);
    }
    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYolov8SegMetalNut);
