/**
 * @file smoothing_filter.h
 * @brief One Euro Filter for real-time gaze coordinate smoothing.
 *
 * Reference: Géry Casiez, Nicolas Roussel, Daniel Vogel.
 * "1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in
 *  Interactive Systems." CHI 2012.
 *
 * Two independent instances are used (one for X, one for Y) to produce
 * smooth, low-latency cursor movement from noisy gaze predictions.
 */

#pragma once

#include <cstdint>

namespace gaze {

class OneEuroFilter {
public:
    /**
     * @param min_cutoff  Minimum cutoff frequency (Hz). Lower = smoother but more lag.
     *                    Typical range: 0.5–2.0. Default: 1.0.
     * @param beta        Speed coefficient. Higher = less lag for fast movements.
     *                    Typical range: 0.0–0.01. Default: 0.007.
     * @param d_cutoff    Cutoff frequency for the derivative filter. Default: 1.0.
     */
    explicit OneEuroFilter(float min_cutoff = 1.0f,
                           float beta = 0.007f,
                           float d_cutoff = 1.0f);

    /**
     * @brief Filter a single value.
     * @param value        Raw input value.
     * @param timestamp_us Timestamp in microseconds (monotonic).
     * @return Smoothed value.
     */
    float filter(float value, int64_t timestamp_us);

    /** @brief Reset internal state. Call when tracking is lost. */
    void reset();

private:
    float min_cutoff_;
    float beta_;
    float d_cutoff_;

    float x_prev_;
    float dx_prev_;
    int64_t t_prev_us_;
    bool initialized_;

    static float smoothing_factor(float te, float cutoff);
    static float exponential_smoothing(float alpha, float x, float x_prev);
};

/**
 * @brief Convenience wrapper for filtering 2D gaze coordinates.
 */
class GazeSmoother {
public:
    GazeSmoother(float min_cutoff = 1.0f, float beta = 0.007f, float d_cutoff = 1.0f);

    struct Point2D { float x, y; };

    /**
     * @brief Smooth a 2D gaze coordinate.
     * @param x, y         Raw gaze coordinates.
     * @param timestamp_us Timestamp in microseconds.
     * @return Smoothed (x, y).
     */
    Point2D filter(float x, float y, int64_t timestamp_us);

    void reset();

private:
    OneEuroFilter filter_x_;
    OneEuroFilter filter_y_;
};

}  // namespace gaze
