package app.eyegaze.domain.filter

import app.eyegaze.domain.geometry.ScreenPoint
import kotlin.math.PI
import kotlin.math.abs

/**
 * One Euro filter (Casiez, Roussel & Vogel, CHI 2012): a low-pass filter whose cutoff rises with speed —
 * steady fixations are smoothed strongly (low jitter), fast saccades pass with little lag.
 *
 * @param minCutoff Hz; lower = smoother fixations, more lag.
 * @param beta speed coefficient (per unit/s); higher = less lag on fast movements.
 * @param dCutoff Hz; cutoff of the derivative estimate.
 */
public class OneEuroFilter(
    private val minCutoff: Double = 1.0,
    private val beta: Double = 0.3,
    private val dCutoff: Double = 1.0,
) {
    init {
        require(minCutoff > 0 && dCutoff > 0) { "Cutoffs must be positive" }
        require(beta >= 0) { "beta must be non-negative" }
    }

    private var lastTimeNanos: Long = 0
    private var lastValue: Double = 0.0
    private var lastDerivative: Double = 0.0
    private var initialized = false

    public fun reset() {
        initialized = false
    }

    /** Filters [value] observed at [timeNanos]; non-increasing timestamps return the previous output. */
    public fun filter(
        value: Double,
        timeNanos: Long,
    ): Double {
        require(value.isFinite()) { "value must be finite" }
        if (!initialized) {
            initialized = true
            lastTimeNanos = timeNanos
            lastValue = value
            lastDerivative = 0.0
            return value
        }
        val dt = (timeNanos - lastTimeNanos) / 1e9
        if (dt <= 0.0) return lastValue
        val derivative = (value - lastValue) / dt
        val smoothedDerivative = lerp(lastDerivative, derivative, alpha(dCutoff, dt))
        val cutoff = minCutoff + beta * abs(smoothedDerivative)
        val result = lerp(lastValue, value, alpha(cutoff, dt))
        lastTimeNanos = timeNanos
        lastValue = result
        lastDerivative = smoothedDerivative
        return result
    }

    private fun alpha(
        cutoff: Double,
        dt: Double,
    ): Double {
        val tau = 1.0 / (2.0 * PI * cutoff)
        return 1.0 / (1.0 + tau / dt)
    }

    private fun lerp(
        a: Double,
        b: Double,
        t: Double,
    ): Double = a + t * (b - a)
}

/** Independent One Euro filters on x and y. */
public class PointFilter(
    minCutoff: Double = 1.0,
    beta: Double = 0.3,
    dCutoff: Double = 1.0,
) {
    private val fx = OneEuroFilter(minCutoff, beta, dCutoff)
    private val fy = OneEuroFilter(minCutoff, beta, dCutoff)

    public fun filter(
        point: ScreenPoint,
        timeNanos: Long,
    ): ScreenPoint = ScreenPoint(fx.filter(point.x, timeNanos), fy.filter(point.y, timeNanos))

    public fun reset() {
        fx.reset()
        fy.reset()
    }
}
