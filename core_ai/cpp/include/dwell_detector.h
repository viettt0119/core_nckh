/**
 * @file dwell_detector.h
 * @brief Ring buffer-based dwell time detector for gaze fixation triggers.
 *
 * Stores the last N gaze coordinates in a circular buffer, computes
 * the spatial variance, and emits a "dwell" signal when the user
 * fixates on a point for longer than the configured threshold.
 */

#pragma once

#include <array>
#include <cstdint>

namespace gaze {

class DwellDetector {
public:
    static constexpr int BUFFER_SIZE = 30;  // ~1 second at 30 fps

    /**
     * @param dwell_time_ms          Minimum fixation time to trigger dwell (ms).
     * @param variance_threshold     Maximum spatial variance to count as fixation.
     */
    explicit DwellDetector(int dwell_time_ms = 800,
                           float variance_threshold = 0.002f);

    /**
     * @brief Update with new gaze coordinates.
     * @param x, y         Current gaze position (0.0–1.0).
     * @param timestamp_ms Current timestamp in milliseconds.
     * @return true if a dwell trigger should fire.
     */
    bool update(float x, float y, int64_t timestamp_ms);

    /** @brief Reset state. Call when face is lost. */
    void reset();

    /** @brief Get the current spatial variance (for debugging). */
    float current_variance() const;

private:
    int dwell_time_ms_;
    float variance_threshold_;

    std::array<float, BUFFER_SIZE> x_buf_;
    std::array<float, BUFFER_SIZE> y_buf_;
    int head_ = 0;
    int count_ = 0;

    int64_t fixation_start_ms_ = 0;
    bool fixating_ = false;
    bool fired_ = false;  // Prevent repeated firing for same fixation

    float compute_variance() const;
};

}  // namespace gaze
