/**
 * @file gaze_core.h
 * @brief Public C API for the EyeGaze native inference engine.
 *
 * This header defines the extern "C" interface used by Flutter (via dart:ffi),
 * Android JNI, or iOS bridging to call the C++ gaze tracking pipeline.
 *
 * Lifecycle:
 *   1. gaze_init(model_path)       — Load TFLite model, allocate resources
 *   2. gaze_process_frame(...)     — Per-frame inference + filtering
 *   3. gaze_destroy()              — Release all resources
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Result struct returned from each processed frame                   */
/* ------------------------------------------------------------------ */
typedef struct {
    float x;            /**< Predicted gaze X coordinate (0.0 = left, 1.0 = right) */
    float y;            /**< Predicted gaze Y coordinate (0.0 = top,  1.0 = bottom) */
    float raw_x;        /**< Unfiltered gaze X (before smoothing) */
    float raw_y;        /**< Unfiltered gaze Y (before smoothing) */
    int   direction;    /**< Discrete direction: 0=left, 1=right, 2=center, 3=up, 4=down */
    float confidence;   /**< Softmax confidence of the direction prediction (0.0–1.0) */
    int   is_dwell;     /**< 1 if gaze has been fixated long enough to trigger a dwell action */
    int   is_blink;     /**< 1 if a blink was detected in this frame */
} GazeResult;

/* ------------------------------------------------------------------ */
/* Configuration                                                       */
/* ------------------------------------------------------------------ */
typedef struct {
    int   image_size;           /**< Eye crop size in pixels (default: 64)  */
    int   face_size;            /**< Face crop size in pixels (default: 96) */
    float filter_min_cutoff;    /**< One Euro Filter: min cutoff freq (default: 1.0)  */
    float filter_beta;          /**< One Euro Filter: speed coefficient (default: 0.007) */
    int   dwell_time_ms;        /**< Dwell trigger time in milliseconds (default: 800) */
    float dwell_variance_threshold; /**< Max variance to count as fixation (default: 0.002) */
} GazeConfig;

/**
 * @brief Get a GazeConfig initialized with sensible defaults.
 */
GazeConfig gaze_default_config(void);

/* ------------------------------------------------------------------ */
/* Lifecycle                                                           */
/* ------------------------------------------------------------------ */

/**
 * @brief Initialize the gaze engine with a TFLite model.
 * @param model_path  Absolute path to the .tflite model file.
 * @param config      Configuration parameters (pass NULL for defaults).
 * @return 0 on success, negative error code on failure.
 */
int gaze_init(const char* model_path, const GazeConfig* config);

/**
 * @brief Process a single camera frame.
 *
 * The frame is expected in NV21 (YUV 420sp) format, which is the default
 * output of Android CameraX and iOS AVCaptureSession.
 *
 * @param yuv_data   Pointer to the NV21 pixel buffer.
 * @param width      Frame width in pixels.
 * @param height     Frame height in pixels.
 * @param rotation   Device rotation in degrees (0, 90, 180, 270).
 * @param face_x     Face bounding box X (from on-device face detector).
 * @param face_y     Face bounding box Y.
 * @param face_w     Face bounding box width.
 * @param face_h     Face bounding box height.
 * @param left_eye_x Left eye bounding box X.
 * @param left_eye_y Left eye bounding box Y.
 * @param left_eye_w Left eye bounding box width.
 * @param left_eye_h Left eye bounding box height.
 * @param right_eye_x Right eye bounding box X.
 * @param right_eye_y Right eye bounding box Y.
 * @param right_eye_w Right eye bounding box width.
 * @param right_eye_h Right eye bounding box height.
 * @param timestamp_ms  Frame timestamp in milliseconds.
 * @return GazeResult with smoothed coordinates and trigger signals.
 */
GazeResult gaze_process_frame(
    const unsigned char* yuv_data,
    int width, int height, int rotation,
    float face_x, float face_y, float face_w, float face_h,
    float left_eye_x, float left_eye_y, float left_eye_w, float left_eye_h,
    float right_eye_x, float right_eye_y, float right_eye_w, float right_eye_h,
    int64_t timestamp_ms
);

/**
 * @brief Reset filter state (call after face lost / tracking interruption).
 */
void gaze_reset(void);

/**
 * @brief Release all resources and unload the model.
 */
void gaze_destroy(void);

#ifdef __cplusplus
}
#endif
