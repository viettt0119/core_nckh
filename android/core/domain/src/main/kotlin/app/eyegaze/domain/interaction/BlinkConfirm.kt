package app.eyegaze.domain.interaction

import app.eyegaze.domain.geometry.ScreenPoint

/**
 * Thresholds for the deliberate-blink confirmation (ADR-0008). Blink scores are MediaPipe blendshapes
 * `eyeBlinkLeft/Right` in [0, 1]; [closeThreshold] > [openThreshold] gives hysteresis.
 */
public data class BlinkConfig(
    val closeThreshold: Double = 0.55,
    val openThreshold: Double = 0.35,
    val armMs: Long = 400,
    val maxMs: Long = 1_500,
    val cooldownMs: Long = 500,
) {
    init {
        require(openThreshold in 0.0..1.0 && closeThreshold in 0.0..1.0) { "Thresholds must be in [0, 1]" }
        require(closeThreshold > openThreshold) { "closeThreshold must exceed openThreshold (hysteresis)" }
        require(armMs in 1 until maxMs) { "Need 0 < armMs < maxMs" }
        require(cooldownMs >= 0) { "cooldownMs must be non-negative" }
    }
}

public enum class BlinkPhase { TRACKING, CLOSING, ARMED, CANCELLED, COOLDOWN }

public sealed interface BlinkEvent {
    /** Both eyes started closing at [timeMs]. */
    public data class ClosingStarted(
        val timeMs: Long,
    ) : BlinkEvent

    /** A deliberate blink (armMs ≤ duration ≤ maxMs) just ended. */
    public data class Confirmed(
        val closedAtMs: Long,
        val durationMs: Long,
    ) : BlinkEvent
}

/**
 * Deliberate-blink detector:
 * TRACKING → CLOSING (both eyes closed) → ARMED (closed ≥ armMs) → confirm on reopening (≤ maxMs) → COOLDOWN.
 * Reopening before armMs is a natural blink and is ignored; staying closed beyond maxMs cancels.
 */
public class BlinkFsm(
    public val config: BlinkConfig = BlinkConfig(),
) {
    public var phase: BlinkPhase = BlinkPhase.TRACKING
        private set
    private var closedAt: Long = 0
    private var cooldownUntil: Long = 0
    private var eyesClosed = false

    /** Progress towards arming in [0, 1], for UI feedback. */
    public fun progress(timeMs: Long): Double =
        when (phase) {
            BlinkPhase.CLOSING -> ((timeMs - closedAt).toDouble() / config.armMs).coerceIn(0.0, 1.0)
            BlinkPhase.ARMED -> 1.0
            else -> 0.0
        }

    public val isEyesClosed: Boolean get() = eyesClosed

    public fun update(
        blinkLeft: Double,
        blinkRight: Double,
        timeMs: Long,
    ): BlinkEvent? {
        require(blinkLeft.isFinite() && blinkRight.isFinite()) { "Blink scores must be finite" }
        if (!eyesClosed && minOf(blinkLeft, blinkRight) > config.closeThreshold) {
            eyesClosed = true
        } else if (eyesClosed && maxOf(blinkLeft, blinkRight) < config.openThreshold) {
            eyesClosed = false
        }
        return when (phase) {
            BlinkPhase.TRACKING ->
                if (eyesClosed) {
                    phase = BlinkPhase.CLOSING
                    closedAt = timeMs
                    BlinkEvent.ClosingStarted(timeMs)
                } else {
                    null
                }
            BlinkPhase.CLOSING -> {
                when {
                    !eyesClosed -> phase = BlinkPhase.TRACKING
                    timeMs - closedAt >= config.armMs -> phase = BlinkPhase.ARMED
                }
                null
            }
            BlinkPhase.ARMED -> {
                val duration = timeMs - closedAt
                when {
                    duration > config.maxMs -> {
                        phase = if (eyesClosed) BlinkPhase.CANCELLED else BlinkPhase.TRACKING
                        null
                    }
                    !eyesClosed -> {
                        phase = BlinkPhase.COOLDOWN
                        cooldownUntil = timeMs + config.cooldownMs
                        BlinkEvent.Confirmed(closedAt, duration)
                    }
                    else -> null
                }
            }
            BlinkPhase.CANCELLED -> {
                if (!eyesClosed) phase = BlinkPhase.TRACKING
                null
            }
            BlinkPhase.COOLDOWN -> {
                if (timeMs >= cooldownUntil && !eyesClosed) phase = BlinkPhase.TRACKING
                null
            }
        }
    }

    public fun reset() {
        phase = BlinkPhase.TRACKING
        eyesClosed = false
    }
}

/** Per-frame input to the selection engine; [gaze] is null when no face is visible. */
public data class FrameSignal(
    val timeMs: Long,
    val gaze: ScreenPoint?,
    val blinkLeft: Double,
    val blinkRight: Double,
)

public data class SelectionState(
    val hoverCell: Int?,
    val phase: BlinkPhase,
    val blinkProgress: Double,
    val lockedCell: Int?,
    /** Non-null on exactly the frame a selection is confirmed. */
    val selectedCell: Int?,
    val faceVisible: Boolean,
)

/**
 * Look-to-select, blink-to-confirm. Gaze only moves the hover while the eyes are open; when a closure starts,
 * the cell hovered [lookbackMs] earlier is locked (gaze drifts as the eyelids drop) and selected if the blink
 * turns out to be deliberate.
 */
public class GazeSelectionEngine(
    public val grid: CellGrid,
    private val selector: CellSelector = CellSelector(),
    private val blink: BlinkFsm = BlinkFsm(),
    private val lookbackMs: Long = 100,
) {
    private var lockedCell: Int? = null

    public fun onFrame(signal: FrameSignal): SelectionState {
        val gaze = signal.gaze
        if (gaze == null) {
            reset()
            return SelectionState(null, BlinkPhase.TRACKING, 0.0, null, null, faceVisible = false)
        }
        val event = blink.update(signal.blinkLeft, signal.blinkRight, signal.timeMs)
        if (!blink.isEyesClosed && (blink.phase == BlinkPhase.TRACKING || blink.phase == BlinkPhase.COOLDOWN)) {
            selector.update(grid.cellAt(gaze), signal.timeMs)
        }
        var selected: Int? = null
        when (event) {
            is BlinkEvent.ClosingStarted -> lockedCell = selector.hoverAt(event.timeMs - lookbackMs) ?: selector.hover
            is BlinkEvent.Confirmed -> {
                selected = lockedCell
                lockedCell = null
            }
            null -> if (blink.phase == BlinkPhase.TRACKING) lockedCell = null
        }
        return SelectionState(
            hoverCell = selector.hover,
            phase = blink.phase,
            blinkProgress = blink.progress(signal.timeMs),
            lockedCell = lockedCell,
            selectedCell = selected,
            faceVisible = true,
        )
    }

    public fun reset() {
        selector.reset()
        blink.reset()
        lockedCell = null
    }
}
