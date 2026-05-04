/**
 * @file tflite_inference.cpp
 * @brief TensorFlow Lite model loading and inference execution.
 */

#include "tflite_inference.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

#ifndef GAZE_NO_TFLITE
#include "tensorflow/lite/kernels/register.h"
#endif

namespace gaze {

TFLiteRunner::TFLiteRunner() = default;
TFLiteRunner::~TFLiteRunner() = default;

bool TFLiteRunner::is_loaded() const {
    return loaded_;
}

bool TFLiteRunner::load_model(const std::string& model_path) {
#ifdef GAZE_NO_TFLITE
    (void)model_path;
    return false;
#else
    model_ = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!model_) {
        return false;
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model_, resolver);
    builder(&interpreter_);

    if (!interpreter_) {
        return false;
    }

    // Use 2 threads for inference (good balance for mobile)
    interpreter_->SetNumThreads(2);

    if (interpreter_->AllocateTensors() != kTfLiteOk) {
        interpreter_.reset();
        return false;
    }

    loaded_ = true;
    return true;
#endif
}

// --------------------------------------------------------------------------
// Softmax helper
// --------------------------------------------------------------------------

static void softmax(const float* input, float* output, int size) {
    float max_val = *std::max_element(input, input + size);
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }
    for (int i = 0; i < size; ++i) {
        output[i] /= sum;
    }
}

// --------------------------------------------------------------------------
// CHW → HWC transposition for TFLite input
// --------------------------------------------------------------------------

/**
 * @brief Transpose a CHW tensor to HWC format.
 *
 * PyTorch uses CHW layout (channels-first), but TFLite models converted
 * from ONNX typically expect HWC layout (channels-last).
 *
 * @param chw   Input data in [C, H, W] layout.
 * @param hwc   Output buffer in [H, W, C] layout (must be pre-allocated).
 * @param c, h, w  Tensor dimensions.
 */
static void chw_to_hwc(const float* chw, float* hwc, int c, int h, int w) {
    for (int ch = 0; ch < c; ++ch) {
        for (int row = 0; row < h; ++row) {
            for (int col = 0; col < w; ++col) {
                // CHW index: ch * h * w + row * w + col
                // HWC index: row * w * c + col * c + ch
                hwc[row * w * c + col * c + ch] = chw[ch * h * w + row * w + col];
            }
        }
    }
}

InferenceResult TFLiteRunner::run(const float* face_data,
                                   const float* left_eye_data,
                                   const float* right_eye_data,
                                   const float* face_grid,
                                   const float* geometry,
                                   int image_size,
                                   int face_size) {
    InferenceResult result{};

#ifdef GAZE_NO_TFLITE
    (void)face_data; (void)left_eye_data; (void)right_eye_data;
    (void)face_grid; (void)geometry; (void)image_size; (void)face_size;
    result.gaze_x = 0.5f;
    result.gaze_y = 0.5f;
    result.direction = 2;  // center
    result.confidence = 0.0f;
    return result;
#else
    if (!loaded_ || !interpreter_) {
        result.gaze_x = 0.5f;
        result.gaze_y = 0.5f;
        result.direction = 2;
        result.confidence = 0.0f;
        return result;
    }

    // --- Copy input tensors ---
    // Input order matches ONNX export: face, left_eye, right_eye, face_grid, geometry
    // TFLite expects HWC, so we transpose from CHW

    // Input 0: face [1, face_size, face_size, 3]
    float* face_input = interpreter_->typed_input_tensor<float>(0);
    if (face_input) {
        std::vector<float> hwc_buf(3 * face_size * face_size);
        chw_to_hwc(face_data, hwc_buf.data(), 3, face_size, face_size);
        std::memcpy(face_input, hwc_buf.data(), hwc_buf.size() * sizeof(float));
    }

    // Input 1: left_eye [1, image_size, image_size, 3]
    float* left_input = interpreter_->typed_input_tensor<float>(1);
    if (left_input) {
        std::vector<float> hwc_buf(3 * image_size * image_size);
        chw_to_hwc(left_eye_data, hwc_buf.data(), 3, image_size, image_size);
        std::memcpy(left_input, hwc_buf.data(), hwc_buf.size() * sizeof(float));
    }

    // Input 2: right_eye [1, image_size, image_size, 3]
    float* right_input = interpreter_->typed_input_tensor<float>(2);
    if (right_input) {
        std::vector<float> hwc_buf(3 * image_size * image_size);
        chw_to_hwc(right_eye_data, hwc_buf.data(), 3, image_size, image_size);
        std::memcpy(right_input, hwc_buf.data(), hwc_buf.size() * sizeof(float));
    }

    // Input 3: face_grid [1, 25, 25]
    float* grid_input = interpreter_->typed_input_tensor<float>(3);
    if (grid_input) {
        std::memcpy(grid_input, face_grid, 25 * 25 * sizeof(float));
    }

    // Input 4: geometry [1, 12]
    float* geom_input = interpreter_->typed_input_tensor<float>(4);
    if (geom_input) {
        std::memcpy(geom_input, geometry, 12 * sizeof(float));
    }

    // --- Run inference ---
    if (interpreter_->Invoke() != kTfLiteOk) {
        result.gaze_x = 0.5f;
        result.gaze_y = 0.5f;
        result.direction = 2;
        result.confidence = 0.0f;
        return result;
    }

    // --- Extract outputs ---
    // Output 0: gaze_xy [1, 2]
    const float* gaze_output = interpreter_->typed_output_tensor<float>(0);
    if (gaze_output) {
        result.gaze_x = gaze_output[0];
        result.gaze_y = gaze_output[1];
    }

    // Output 1: direction_logits [1, 5]
    const float* dir_output = interpreter_->typed_output_tensor<float>(1);
    if (dir_output) {
        softmax(dir_output, result.direction_probs.data(), 5);
        result.direction = static_cast<int>(
            std::distance(result.direction_probs.begin(),
                          std::max_element(result.direction_probs.begin(),
                                           result.direction_probs.end()))
        );
        result.confidence = result.direction_probs[result.direction];
    }

    return result;
#endif
}

}  // namespace gaze
