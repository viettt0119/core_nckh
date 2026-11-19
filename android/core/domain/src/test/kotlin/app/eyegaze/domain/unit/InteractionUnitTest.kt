package app.eyegaze.domain.unit

import app.eyegaze.domain.filter.OneEuroFilter
import app.eyegaze.domain.geometry.ScreenPoint
import app.eyegaze.domain.interaction.BlinkConfig
import app.eyegaze.domain.interaction.BlinkEvent
import app.eyegaze.domain.interaction.BlinkFsm
import app.eyegaze.domain.interaction.BlinkPhase
import app.eyegaze.domain.interaction.CellGrid
import app.eyegaze.domain.interaction.CellSelector
import app.eyegaze.domain.interaction.FrameSignal
import app.eyegaze.domain.interaction.GazeSelectionEngine
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import kotlin.math.abs
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertNull
import kotlin.test.assertTrue

private const val FRAME_MS = 33L

class OneEuroFilterTest {
    @Test
    fun `constant input passes unchanged and steps converge without overshoot`() {
        val f = OneEuroFilter()
        repeat(10) { assertEquals(0.5, f.filter(0.5, it * 33_000_000L), 1e-12) }
        var last = 0.5
        for (k in 10 until 200) {
            val y = f.filter(1.0, k * 33_000_000L)
            assertTrue(y >= last - 1e-12 && y <= 1.0 + 1e-12)
            last = y
        }
        assertTrue(abs(last - 1.0) < 1e-3)
    }

    @Test
    fun `jitter is attenuated and stale timestamps are ignored`() {
        val f = OneEuroFilter(minCutoff = 1.0, beta = 0.0)
        var maxDev = 0.0
        for (k in 0 until 300) {
            val noisy = 0.5 + if (k % 2 == 0) 0.02 else -0.02
            val y = f.filter(noisy, k * 33_000_000L)
            if (k > 30) maxDev = maxOf(maxDev, abs(y - 0.5))
        }
        assertTrue(maxDev < 0.01, "residual jitter $maxDev")
        val before = f.filter(0.9, 300 * 33_000_000L)
        assertEquals(before, f.filter(0.1, 300 * 33_000_000L))
        f.reset()
        assertEquals(0.1, f.filter(0.1, 0))
    }
}

class CellGridTest {
    @Test
    fun `cells are row major and off screen is null`() {
        val grid = CellGrid(3, 4)
        assertEquals(0, grid.cellAt(ScreenPoint(0.0, 0.0)))
        assertEquals(11, grid.cellAt(ScreenPoint(0.99, 0.99)))
        assertEquals(5, grid.cellAt(ScreenPoint(0.3, 0.5)))
        assertNull(grid.cellAt(ScreenPoint(1.0, 0.5)))
        assertNull(grid.cellAt(ScreenPoint(-0.01, 0.5)))
        assertEquals(5, grid.cellAt(grid.center(5)))
        assertThrows<IllegalArgumentException> { CellGrid(0, 3) }
    }
}

class CellSelectorTest {
    @Test
    fun `hover changes only after dwell and remembers history`() {
        val s = CellSelector(dwellMs = 250)
        assertNull(s.update(2, 0))
        assertEquals(2, s.update(2, 250))
        assertEquals(2, s.update(3, 300)) // brief excursion…
        assertEquals(2, s.update(2, 400)) // …is ignored
        assertEquals(2, s.update(5, 500))
        assertEquals(5, s.update(5, 760))
        assertEquals(2, s.hoverAt(700))
        assertEquals(5, s.hoverAt(800))
        assertNull(s.hoverAt(100))
        s.reset()
        assertNull(s.hover)
    }
}

class BlinkFsmTest {
    private fun run(
        fsm: BlinkFsm,
        closedFrom: Long,
        closedUntil: Long,
        endMs: Long,
    ): List<Pair<Long, BlinkEvent>> {
        val events = mutableListOf<Pair<Long, BlinkEvent>>()
        var t = 0L
        while (t <= endMs) {
            val closed = t in closedFrom until closedUntil
            fsm.update(if (closed) 0.9 else 0.05, if (closed) 0.9 else 0.05, t)?.let { events += t to it }
            t += FRAME_MS
        }
        return events
    }

    @Test
    fun `natural blink is ignored`() {
        val events = run(BlinkFsm(), 330, 530, 1500)
        assertEquals(1, events.size)
        assertIs<BlinkEvent.ClosingStarted>(events[0].second)
    }

    @Test
    fun `deliberate blink confirms once then cools down`() {
        val fsm = BlinkFsm()
        val events = run(fsm, 330, 1000, 1200)
        val confirmed = events.map { it.second }.filterIsInstance<BlinkEvent.Confirmed>()
        assertEquals(1, confirmed.size)
        assertTrue(confirmed[0].durationMs in 400..1500)
        assertEquals(BlinkPhase.COOLDOWN, fsm.phase)
    }

    @Test
    fun `too long a closure cancels`() {
        val fsm = BlinkFsm()
        val events = run(fsm, 330, 2500, 3000)
        assertTrue(events.none { it.second is BlinkEvent.Confirmed })
        assertEquals(BlinkPhase.TRACKING, fsm.phase)
    }

    @Test
    fun `one eye closed is not a blink and hysteresis holds`() {
        val fsm = BlinkFsm()
        for (t in 0L..2000L step FRAME_MS) assertNull(fsm.update(0.95, 0.1, t))
        assertEquals(BlinkPhase.TRACKING, fsm.phase)
        fsm.update(0.9, 0.9, 2100)
        fsm.update(0.45, 0.45, 2133) // between thresholds: still closed
        assertTrue(fsm.isEyesClosed)
        assertThrows<IllegalArgumentException> { BlinkConfig(closeThreshold = 0.3, openThreshold = 0.4) }
    }
}

class GazeSelectionEngineTest {
    private val grid = CellGrid(2, 2)
    private val topLeft = ScreenPoint(0.25, 0.25)
    private val bottomRight = ScreenPoint(0.75, 0.75)

    @Test
    fun `look then deliberate blink selects the locked cell even if gaze drifts while closing`() {
        val engine = GazeSelectionEngine(grid)
        var t = 0L
        var selected: Int? = null
        while (t < 600) {
            engine.onFrame(FrameSignal(t, topLeft, 0.05, 0.05))
            t += FRAME_MS
        }
        assertEquals(0, engine.onFrame(FrameSignal(t, topLeft, 0.05, 0.05)).hoverCell)
        val closeStart = t + FRAME_MS
        t = closeStart
        while (t < closeStart + 700) {
            val s = engine.onFrame(FrameSignal(t, bottomRight, 0.9, 0.9)) // gaze estimate drifts while eyes close
            assertEquals(0, s.lockedCell)
            t += FRAME_MS
        }
        val opened = engine.onFrame(FrameSignal(t, topLeft, 0.05, 0.05))
        selected = opened.selectedCell
        assertEquals(0, selected)
        assertNull(engine.onFrame(FrameSignal(t + FRAME_MS, topLeft, 0.05, 0.05)).selectedCell)
    }

    @Test
    fun `natural blink selects nothing and losing the face resets`() {
        val engine = GazeSelectionEngine(grid)
        var t = 0L
        while (t < 600) {
            engine.onFrame(FrameSignal(t, bottomRight, 0.05, 0.05))
            t += FRAME_MS
        }
        repeat(5) {
            engine.onFrame(FrameSignal(t, bottomRight, 0.9, 0.9))
            t += FRAME_MS
        }
        val after = engine.onFrame(FrameSignal(t, bottomRight, 0.05, 0.05))
        assertNull(after.selectedCell)
        assertEquals(3, after.hoverCell)
        val lost = engine.onFrame(FrameSignal(t + FRAME_MS, null, 0.0, 0.0))
        assertEquals(false, lost.faceVisible)
        assertNull(engine.onFrame(FrameSignal(t + 2 * FRAME_MS, bottomRight, 0.05, 0.05)).hoverCell)
    }
}
