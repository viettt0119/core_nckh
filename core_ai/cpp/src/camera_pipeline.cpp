/**
 * @file camera_pipeline.cpp
 * @brief YUV→RGB conversion and image preprocessing — 100% parity with Python.
 *
 * Every OpenCV call here mirrors src/eyegaze/preprocess.py exactly.
 * See the flags reference block at the top of preprocess.py for the mapping.
 */

#include "camera_pipeline.h"

#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>

namespace gaze {

// --------------------------------------------------------------------------
// YUV → BGR conversion with rotation
// --------------------------------------------------------------------------

cv::Mat yuv_nv21_to_bgr(const unsigned char* yuv_data,
                         int width, int height, int rotation_degrees) {
    // NV21 buffer size = width * height * 1.5
    cv::Mat yuv(height + height / 2, width, CV_8UC1, const_cast<unsigned char*>(yuv_data));
    cv::Mat bgr;
    cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV21);

    // Apply device rotation
    switch (rotation_degrees) {
        case 90:
            cv::rotate(bgr, bgr, cv::ROTATE_90_CLOCKWISE);
            break;
        case 180:
            cv::rotate(bgr, bgr, cv::ROTATE_180);
            break;
        case 270:
            cv::rotate(bgr, bgr, cv::ROTATE_90_COUNTERCLOCKWISE);
            break;
        default:
            break;  // 0 degrees — no rotation
    }

    return bgr;
}

// --------------------------------------------------------------------------
// crop_resize — mirrors Python preprocess.crop_resize() exactly
// --------------------------------------------------------------------------

cv::Mat crop_resize(const cv::Mat& image_bgr,
                    float x, float y, float w, float h,
                    int output_size, bool flip_horizontal) {
    int img_h = image_bgr.rows;
    int img_w = image_bgr.cols;

    // Clamp box to image bounds (mirrors Python clamp_box)
    int x0 = std::max(0, std::min(img_w, static_cast<int>(std::round(x))));
    int y0 = std::max(0, std::min(img_h, static_cast<int>(std::round(y))));
    int x1 = std::max(0, std::min(img_w, static_cast<int>(std::round(x + w))));
    int y1 = std::max(0, std::min(img_h, static_cast<int>(std::round(y + h))));

    // Return zero tensor if box is invalid
    if (x1 <= x0 || y1 <= y0) {
        return cv::Mat::zeros(output_size, output_size, CV_32FC3);
    }

    // Crop
    cv::Mat crop = image_bgr(cv::Rect(x0, y0, x1 - x0, y1 - y0));
    if (crop.empty()) {
        return cv::Mat::zeros(output_size, output_size, CV_32FC3);
    }

    // Optional horizontal flip (for left eye — mirrors Python cv2.flip(crop, 1))
    if (flip_horizontal) {
        cv::flip(crop, crop, 1);
    }

    // BGR → RGB (cv::COLOR_BGR2RGB = 4, mirrors Python cv2.COLOR_BGR2RGB)
    cv::Mat rgb;
    cv::cvtColor(crop, rgb, cv::COLOR_BGR2RGB);

    // Resize with INTER_AREA (= 3, mirrors Python cv2.INTER_AREA)
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(output_size, output_size), 0, 0, cv::INTER_AREA);

    // Normalize to [0, 1] float32 (mirrors Python crop.astype(np.float32) / 255.0)
    cv::Mat normalized;
    resized.convertTo(normalized, CV_32FC3, 1.0 / 255.0);

    return normalized;
}

// --------------------------------------------------------------------------
// face_grid_from_box — mirrors Python preprocess.face_grid_from_box()
// --------------------------------------------------------------------------

std::array<float, 625> face_grid_from_box(int image_width, int image_height,
                                           float face_x, float face_y,
                                           float face_w, float face_h) {
    constexpr int GRID_SIZE = 25;
    std::array<float, 625> grid{};
    grid.fill(0.0f);

    float w = static_cast<float>(std::max(image_width, 1));
    float h = static_cast<float>(std::max(image_height, 1));

    // Mirrors Python: np.floor(x / width * grid_size)
    int gx0 = std::max(0, std::min(GRID_SIZE, static_cast<int>(std::floor(face_x / w * GRID_SIZE))));
    int gy0 = std::max(0, std::min(GRID_SIZE, static_cast<int>(std::floor(face_y / h * GRID_SIZE))));
    // Mirrors Python: np.ceil((x + w) / width * grid_size)
    int gx1 = std::max(0, std::min(GRID_SIZE, static_cast<int>(std::ceil((face_x + face_w) / w * GRID_SIZE))));
    int gy1 = std::max(0, std::min(GRID_SIZE, static_cast<int>(std::ceil((face_y + face_h) / h * GRID_SIZE))));

    if (gx1 > gx0 && gy1 > gy0) {
        for (int row = gy0; row < gy1; ++row) {
            for (int col = gx0; col < gx1; ++col) {
                grid[row * GRID_SIZE + col] = 1.0f;
            }
        }
    }

    return grid;
}

// --------------------------------------------------------------------------
// geometry_vector — mirrors Python preprocess.geometry_vector()
// --------------------------------------------------------------------------

std::array<float, 12> geometry_vector(int image_width, int image_height,
                                       float fx, float fy, float fw, float fh,
                                       float lx, float ly, float lw, float lh,
                                       float rx, float ry, float rw, float rh) {
    float sw = static_cast<float>(std::max(image_width, 1));
    float sh = static_cast<float>(std::max(image_height, 1));

    // Scale array: [w, h, w, h] repeated 3 times (for face, left_eye, right_eye)
    float scale[12] = {sw, sh, sw, sh, sw, sh, sw, sh, sw, sh, sw, sh};
    float raw[12] = {fx, fy, fw, fh, lx, ly, lw, lh, rx, ry, rw, rh};

    std::array<float, 12> result{};
    for (int i = 0; i < 12; ++i) {
        float val = raw[i] / scale[i];
        result[i] = std::max(0.0f, std::min(1.5f, val));
    }

    return result;
}

}  // namespace gaze
