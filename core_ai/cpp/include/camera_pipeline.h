/**
 * @file camera_pipeline.h
 * @brief YUV→RGB conversion and image crop/resize pipeline.
 *
 * Mirrors the Python preprocessing in src/eyegaze/preprocess.py exactly:
 *   - cv::COLOR_BGR2RGB (= 4) after YUV→BGR conversion
 *   - cv::INTER_AREA (= 3) for resize
 *   - cv::flip(img, dst, 1) for left eye horizontal flip
 *   - Normalize: float32 = pixel / 255.0f
 *   - Face grid: 25×25 binary matrix using std::floor / std::ceil
 */

#pragma once

#include <array>
#include <vector>
#include <opencv2/core.hpp>

namespace gaze {

/**
 * @brief Convert NV21 (YUV 420sp) buffer to BGR cv::Mat, applying rotation.
 */
cv::Mat yuv_nv21_to_bgr(const unsigned char* yuv_data,
                         int width, int height, int rotation_degrees);

/**
 * @brief Crop a region from the image, resize, convert BGR→RGB, normalize to [0,1].
 *
 * Exact parity with Python crop_resize():
 *   - cv::INTER_AREA interpolation
 *   - cv::COLOR_BGR2RGB
 *   - Optional horizontal flip (for left eye)
 *   - Output: float32 HWC tensor in [0, 1]
 *
 * @param image_bgr     Source BGR image.
 * @param x, y, w, h    Bounding box in pixel coordinates.
 * @param output_size   Target square size (e.g. 64 or 96).
 * @param flip_horizontal  If true, flip the crop horizontally.
 * @return  cv::Mat of shape (output_size, output_size, 3) float32 RGB in [0,1].
 */
cv::Mat crop_resize(const cv::Mat& image_bgr,
                    float x, float y, float w, float h,
                    int output_size, bool flip_horizontal = false);

/**
 * @brief Build a 25×25 binary face grid from a face bounding box.
 *
 * Mirrors Python face_grid_from_box() using std::floor / std::ceil.
 *
 * @param image_width, image_height   Full frame dimensions.
 * @param face_x, face_y, face_w, face_h   Face bounding box.
 * @return  std::array<float, 625> flattened 25×25 grid (row-major).
 */
std::array<float, 625> face_grid_from_box(int image_width, int image_height,
                                           float face_x, float face_y,
                                           float face_w, float face_h);

/**
 * @brief Compute the 12-element geometry vector.
 *
 * Mirrors Python geometry_vector(): normalized [face_box, left_eye_box, right_eye_box].
 */
std::array<float, 12> geometry_vector(int image_width, int image_height,
                                       float fx, float fy, float fw, float fh,
                                       float lx, float ly, float lw, float lh,
                                       float rx, float ry, float rw, float rh);

}  // namespace gaze
