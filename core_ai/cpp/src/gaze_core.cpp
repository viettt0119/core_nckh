/**
 * @file gaze_core.cpp
 * @brief extern "C" API implementation — glue code connecting all modules.
 *
 * This is the single entry point called from Flutter (dart:ffi),
 * Android JNI, or iOS bridging.
 *
 * Pipeline per frame:
 *   1. yuv_nv21_to_bgr()      — Camera buffer → BGR image
 *   2. crop_resize() × 3      — Face, left eye, right eye crops
 *   3. face_grid_from_box()    — 25×25 binary grid
 *   4. geometry_vector()       — 12-element normalized vector
 *   5. TFLiteRunner::run()     — Model inference
 *   6. GazeSmoother::filter()  — One Euro Filter smoothing
 *   7. DwellDetector::update() — Dwell time trigger
 */

#include "gaze_core.h"

#include "camera_pipeline.h"
#include "dwell_detector.h"
#include "smoothing_filter.h"
#include "tflite_inference.h"

#include <opencv2/core.hpp>
#include <cstring>
#include <memory>

// --------------------------------------------------------------------------
// Global state (single-instance design for mobile)
// --------------------------------------------------------------------------

namespace {

struct GazeEngine {
    gaze::TFLiteRunner     runner;
    gaze::GazeSmoother     smoother;
    gaze::DwellDetector    dwell;
    GazeConfig             config;
    bool                   initialized = false;
};

std::unique_ptr<GazeEngine> g_engine;

}  // anonymous namespace

// --------------------------------------------------------------------------
// C API implementation
// --------------------------------------------------------------------------

extern "C" {

GazeConfig gaze_default_config(void) {
    GazeConfig cfg{};
    cfg.image_size              = 64;
    cfg.face_size               = 96;
    cfg.filter_min_cutoff       = 1.0f;
    cfg.filter_beta             = 0.007f;
    cfg.dwell_time_ms           = 800;
    cfg.dwell_variance_threshold = 0.002f;
    return cfg;
}

int gaze_init(const char* model_path, const GazeConfig* config) {
    // Clean up previous instance if any
    g_engine.reset();
    g_engine = std::make_unique<GazeEngine>();

    // Apply config
    GazeConfig cfg = config ? *config : gaze_default_config();
    g_engine->config = cfg;

    // Initialize smoother with config parameters
    g_engine->smoother = gaze::GazeSmoother(
        cfg.filter_min_cutoff, cfg.filter_beta, 1.0f
    );

    // Initialize dwell detector with config parameters
    g_engine->dwell = gaze::DwellDetector(
        cfg.dwell_time_ms, cfg.dwell_variance_threshold
    );

    // Load TFLite model
    if (!g_engine->runner.load_model(model_path)) {
        g_engine.reset();
        return -1;  // Model load failed
    }

    g_engine->initialized = true;
    return 0;
}

GazeResult gaze_process_frame(
    const unsigned char* yuv_data,
    int width, int height, int rotation,
    float face_x, float face_y, float face_w, float face_h,
    float left_eye_x, float left_eye_y, float left_eye_w, float left_eye_h,
    float right_eye_x, float right_eye_y, float right_eye_w, float right_eye_h,
    int64_t timestamp_ms
) {
    GazeResult result{};
    result.x = 0.5f;
    result.y = 0.5f;
    result.raw_x = 0.5f;
    result.raw_y = 0.5f;
    result.direction = 2;  // center
    result.confidence = 0.0f;
    result.is_dwell = 0;
    result.is_blink = 0;

    if (!g_engine || !g_engine->initialized) {
        return result;
    }

    int img_size = g_engine->config.image_size;
    int face_sz  = g_engine->config.face_size;

    // Step 1: YUV → BGR
    cv::Mat bgr = gaze::yuv_nv21_to_bgr(yuv_data, width, height, rotation);

    // Step 2: Crop and preprocess
    //   Face crop (no flip)
    cv::Mat face_crop = gaze::crop_resize(bgr, face_x, face_y, face_w, face_h, face_sz, false);
    //   Left eye crop (with horizontal flip — mirrors Python pipeline)
    cv::Mat left_crop = gaze::crop_resize(bgr, left_eye_x, left_eye_y, left_eye_w, left_eye_h, img_size, true);
    //   Right eye crop (no flip)
    cv::Mat right_crop = gaze::crop_resize(bgr, right_eye_x, right_eye_y, right_eye_w, right_eye_h, img_size, false);

    // Step 3: Face grid (25×25)
    auto grid = gaze::face_grid_from_box(
        bgr.cols, bgr.rows, face_x, face_y, face_w, face_h
    );

    // Step 4: Geometry vector (12 elements)
    auto geom = gaze::geometry_vector(
        bgr.cols, bgr.rows,
        face_x, face_y, face_w, face_h,
        left_eye_x, left_eye_y, left_eye_w, left_eye_h,
        right_eye_x, right_eye_y, right_eye_w, right_eye_h
    );

    // Step 5: Run TFLite inference
    //   crop_resize returns HWC float32, but TFLiteRunner expects CHW.
    //   We pass HWC data directly since the TFLite model (from ONNX)
    //   is already in HWC format after onnx2tf conversion.
    auto inference = g_engine->runner.run(
        reinterpret_cast<const float*>(face_crop.data),
        reinterpret_cast<const float*>(left_crop.data),
        reinterpret_cast<const float*>(right_crop.data),
        grid.data(),
        geom.data(),
        img_size,
        face_sz
    );

    result.raw_x = inference.gaze_x;
    result.raw_y = inference.gaze_y;
    result.direction = inference.direction;
    result.confidence = inference.confidence;

    // Step 6: Smooth coordinates with One Euro Filter
    int64_t timestamp_us = timestamp_ms * 1000;
    auto smoothed = g_engine->smoother.filter(
        inference.gaze_x, inference.gaze_y, timestamp_us
    );
    result.x = smoothed.x;
    result.y = smoothed.y;

    // Step 7: Check dwell trigger
    result.is_dwell = g_engine->dwell.update(smoothed.x, smoothed.y, timestamp_ms) ? 1 : 0;

    // Note: Blink detection is handled on the Flutter/platform side
    // (via face landmark EAR), not in the C++ core.
    result.is_blink = 0;

    return result;
}

void gaze_reset(void) {
    if (g_engine) {
        g_engine->smoother.reset();
        g_engine->dwell.reset();
    }
}

void gaze_destroy(void) {
    g_engine.reset();
}

}  // extern "C"
