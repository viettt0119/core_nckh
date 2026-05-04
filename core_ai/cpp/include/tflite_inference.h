/**
 * @file tflite_inference.h
 * @brief TensorFlow Lite model loader and inference runner.
 */

#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#ifndef GAZE_NO_TFLITE
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#endif

namespace gaze {

struct InferenceResult {
    float gaze_x;           ///< Predicted gaze X (0.0–1.0)
    float gaze_y;           ///< Predicted gaze Y (0.0–1.0)
    int   direction;        ///< Argmax of direction logits (0–4)
    float confidence;       ///< Softmax probability of predicted direction
    std::array<float, 5> direction_probs;  ///< Full softmax distribution
};

class TFLiteRunner {
public:
    TFLiteRunner();
    ~TFLiteRunner();

    /**
     * @brief Load a .tflite model from disk.
     * @return true on success.
     */
    bool load_model(const std::string& model_path);

    /**
     * @brief Run inference on preprocessed inputs.
     *
     * All image inputs are expected as float32 arrays in CHW layout
     * (matching PyTorch convention). The runner handles CHW→HWC
     * transposition internally if the TFLite model expects HWC.
     *
     * @param face_data       Face crop: 3 × face_size × face_size
     * @param left_eye_data   Left eye crop: 3 × image_size × image_size
     * @param right_eye_data  Right eye crop: 3 × image_size × image_size
     * @param face_grid       Face grid: 25 × 25
     * @param geometry        Geometry vector: 12
     * @param image_size      Eye crop resolution (default 64)
     * @param face_size       Face crop resolution (default 96)
     * @return InferenceResult with predicted gaze and direction.
     */
    InferenceResult run(const float* face_data,
                        const float* left_eye_data,
                        const float* right_eye_data,
                        const float* face_grid,
                        const float* geometry,
                        int image_size = 64,
                        int face_size = 96);

    bool is_loaded() const;

private:
#ifndef GAZE_NO_TFLITE
    std::unique_ptr<tflite::FlatBufferModel> model_;
    std::unique_ptr<tflite::Interpreter> interpreter_;
#endif
    bool loaded_ = false;
};

}  // namespace gaze
