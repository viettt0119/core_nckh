package app.eyegaze.domain.interaction

import app.eyegaze.domain.geometry.ScreenPoint

/** A rows × cols grid covering the whole screen; cells are numbered row-major from 0. */
public data class CellGrid(
    val rows: Int,
    val cols: Int,
) {
    init {
        require(rows in 1..12 && cols in 1..12) { "Grid must be between 1x1 and 12x12" }
    }

    val size: Int get() = rows * cols

    /** Cell under a normalized screen point, or null when the point is off-screen. */
    public fun cellAt(p: ScreenPoint): Int? {
        if (!(p.x >= 0.0 && p.x < 1.0 && p.y >= 0.0 && p.y < 1.0)) return null
        return (p.y * rows).toInt() * cols + (p.x * cols).toInt()
    }

    /** Centre of [cell] in normalized screen coordinates. */
    public fun center(cell: Int): ScreenPoint {
        require(cell in 0 until size) { "Cell $cell outside grid" }
        return ScreenPoint((cell % cols + 0.5) / cols, (cell / cols + 0.5) / rows)
    }
}

/**
 * Hover cell with hysteresis: the gaze must stay in a new cell (or off-grid) for [dwellMs] before the hover
 * changes, so jitter at cell borders does not flicker. Keeps a short history so the hover at a past instant
 * can be recovered (the blink-confirm logic locks the cell from just *before* the eyelids started closing).
 */
public class CellSelector(
    private val dwellMs: Long = 250,
    private val historyMs: Long = 2_000,
) {
    init {
        require(dwellMs >= 0 && historyMs > 0) { "Durations must be non-negative" }
    }

    public var hover: Int? = null
        private set
    private var candidate: Int? = null
    private var candidateSince: Long = 0
    private var started = false
    private val history = ArrayDeque<Pair<Long, Int?>>()

    public fun update(
        cell: Int?,
        timeMs: Long,
    ): Int? {
        if (!started || cell != candidate) {
            started = true
            candidate = cell
            candidateSince = timeMs
        }
        if (candidate != hover && timeMs - candidateSince >= dwellMs) {
            hover = candidate
            history.addLast(timeMs to hover)
        }
        while (history.size > 1 && timeMs - history[1].first > historyMs) history.removeFirst()
        return hover
    }

    /** Hover cell as it was at [timeMs] (null before any hover was established). */
    public fun hoverAt(timeMs: Long): Int? = history.lastOrNull { it.first <= timeMs }?.second

    public fun reset() {
        hover = null
        candidate = null
        started = false
        history.clear()
    }
}
