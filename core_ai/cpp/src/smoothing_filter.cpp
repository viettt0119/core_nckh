/**
 * @file smoothing_filter.cpp
 * @brief One Euro Filter implementation for gaze coordinate smoothing.
 *
 * Reference: Géry Casiez et al., "1€ Filter", CHI 2012.
 */

#include "smoothing_filter.h"

#include <cmath>

namespace gaze {

// --------------------------------------------------------------------------
// OneEuroFilter
// --------------------------------------------------------------------------

OneEuroFilter::OneEuroFilter(float min_cutoff, float beta, float d_cutoff)
    : min_cutoff_(min_cutoff)
    , beta_(beta)
    , d_cutoff_(d_cutoff)
    , x_prev_(0.0f)
    , dx_prev_(0.0f)
    , t_prev_us_(0)
    , initialized_(false) {}

float OneEuroFilter::smoothing_factor(float te, float cutoff) {
    float r = 2.0f * static_cast<float>(M_PI) * cutoff * te;
    return r / (r + 1.0f);
}

float OneEuroFilter::exponential_smoothing(float alpha, float x, float x_prev) {
    return alpha * x + (1.0f - alpha) * x_prev;
}

float OneEuroFilter::filter(float value, int64_t timestamp_us) {
    if (!initialized_) {
        x_prev_ = value;
        dx_prev_ = 0.0f;
        t_prev_us_ = timestamp_us;
        initialized_ = true;
        return value;
    }

    // Time elapsed in seconds
    float te = static_cast<float>(timestamp_us - t_prev_us_) / 1'000'000.0f;

    // Guard against zero or negative dt
    if (te <= 0.0f) {
        te = 1.0f / 30.0f;  // Assume 30 fps
    }

    // Estimate derivative
    float dx = (value - x_prev_) / te;
    float alpha_d = smoothing_factor(te, d_cutoff_);
    float dx_hat = exponential_smoothing(alpha_d, dx, dx_prev_);

    // Adaptive cutoff frequency based on speed
    float cutoff = min_cutoff_ + beta_ * std::abs(dx_hat);

    // Filter the value
    float alpha = smoothing_factor(te, cutoff);
    float x_hat = exponential_smoothing(alpha, value, x_prev_);

    // Update state
    x_prev_ = x_hat;
    dx_prev_ = dx_hat;
    t_prev_us_ = timestamp_us;

    return x_hat;
}

void OneEuroFilter::reset() {
    initialized_ = false;
    x_prev_ = 0.0f;
    dx_prev_ = 0.0f;
    t_prev_us_ = 0;
}

// --------------------------------------------------------------------------
// GazeSmoother
// --------------------------------------------------------------------------

GazeSmoother::GazeSmoother(float min_cutoff, float beta, float d_cutoff)
    : filter_x_(min_cutoff, beta, d_cutoff)
    , filter_y_(min_cutoff, beta, d_cutoff) {}

GazeSmoother::Point2D GazeSmoother::filter(float x, float y, int64_t timestamp_us) {
    return {filter_x_.filter(x, timestamp_us), filter_y_.filter(y, timestamp_us)};
}

void GazeSmoother::reset() {
    filter_x_.reset();
    filter_y_.reset();
}

}  // namespace gaze
