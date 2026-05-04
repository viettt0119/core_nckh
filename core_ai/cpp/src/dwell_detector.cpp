/**
 * @file dwell_detector.cpp
 * @brief Ring buffer-based dwell time detector.
 *
 * Stores the last 30 gaze coordinates (~1s at 30fps), computes spatial
 * variance, and triggers a "dwell" signal when variance stays below
 * threshold for the configured duration.
 */

#include "dwell_detector.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace gaze {

DwellDetector::DwellDetector(int dwell_time_ms, float variance_threshold)
    : dwell_time_ms_(dwell_time_ms)
    , variance_threshold_(variance_threshold) {
    x_buf_.fill(0.0f);
    y_buf_.fill(0.0f);
}

float DwellDetector::compute_variance() const {
    if (count_ < 2) {
        return 1.0f;  // Not enough data — return high variance
    }

    int n = std::min(count_, BUFFER_SIZE);

    // Compute mean
    float sum_x = 0.0f, sum_y = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum_x += x_buf_[i];
        sum_y += y_buf_[i];
    }
    float mean_x = sum_x / static_cast<float>(n);
    float mean_y = sum_y / static_cast<float>(n);

    // Compute variance (sum of squared deviations)
    float var = 0.0f;
    for (int i = 0; i < n; ++i) {
        float dx = x_buf_[i] - mean_x;
        float dy = y_buf_[i] - mean_y;
        var += dx * dx + dy * dy;
    }
    var /= static_cast<float>(n);

    return var;
}

bool DwellDetector::update(float x, float y, int64_t timestamp_ms) {
    // Insert into ring buffer
    x_buf_[head_] = x;
    y_buf_[head_] = y;
    head_ = (head_ + 1) % BUFFER_SIZE;
    if (count_ < BUFFER_SIZE) {
        count_++;
    }

    float var = compute_variance();

    if (var < variance_threshold_) {
        // Gaze is stable
        if (!fixating_) {
            fixating_ = true;
            fixation_start_ms_ = timestamp_ms;
            fired_ = false;
        }

        // Check if we've been fixating long enough
        int64_t duration = timestamp_ms - fixation_start_ms_;
        if (!fired_ && duration >= dwell_time_ms_) {
            fired_ = true;
            return true;  // Dwell trigger!
        }
    } else {
        // Gaze is moving — reset fixation tracking
        fixating_ = false;
        fired_ = false;
    }

    return false;
}

void DwellDetector::reset() {
    head_ = 0;
    count_ = 0;
    fixating_ = false;
    fired_ = false;
    fixation_start_ms_ = 0;
    x_buf_.fill(0.0f);
    y_buf_.fill(0.0f);
}

float DwellDetector::current_variance() const {
    return compute_variance();
}

}  // namespace gaze
