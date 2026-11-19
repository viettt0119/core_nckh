package app.eyegaze.board

import app.eyegaze.domain.bundle.BundleException
import app.eyegaze.domain.interaction.BlinkPhase
import app.eyegaze.domain.interaction.CellGrid
import app.eyegaze.domain.interaction.SelectionState
import app.eyegaze.domain.pipeline.CameraFrame
import app.eyegaze.domain.pipeline.GazeFrameResult
import app.eyegaze.domain.pipeline.PipelineMetrics
import app.eyegaze.domain.preprocess.RgbImage
import app.eyegaze.runtime.GazeRuntime
import app.eyegaze.runtime.RuntimeInfo
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertTrue

@OptIn(ExperimentalCoroutinesApi::class)
class BoardViewModelTest {
    private val frame = CameraFrame(RgbImage(1, 1, ByteArray(3)), 0)
    private val info = RuntimeInfo("base", 3, "CPU", "ab".repeat(32), CellGrid(3, 2))

    private class FakeRuntime(
        override val info: RuntimeInfo,
        private val results: ArrayDeque<Any>,
    ) : GazeRuntime {
        var closed = false

        override fun process(frame: CameraFrame): GazeFrameResult =
            when (val next = results.removeFirst()) {
                is Exception -> throw next
                else -> next as GazeFrameResult
            }

        override fun close() {
            closed = true
        }
    }

    private fun result(selected: Int?) =
        GazeFrameResult(
            gaze = null,
            rawGaze = null,
            pose = null,
            selection = SelectionState(1, BlinkPhase.TRACKING, 0.0, null, selected, faceVisible = true),
            metrics = PipelineMetrics(30.0, 10.0, 5.0, 16.0),
        )

    @Test
    fun missingBundleShowsNoModel() {
        val vm = BoardViewModel({ throw BundleException("No model bundle installed") }, UnconfinedTestDispatcher())
        val state = vm.state.value
        assertIs<BoardUiState.NoModel>(state)
        assertTrue("No model" in state.reason)
    }

    @Test
    fun unexpectedStartupErrorShowsFailed() {
        val vm = BoardViewModel({ throw IllegalStateException("boom") }, UnconfinedTestDispatcher())
        assertEquals(BoardUiState.Failed("boom"), vm.state.value)
    }

    @Test
    fun framesUpdateStateCountSelectionsAndSurviveErrors() {
        val runtime = FakeRuntime(info, ArrayDeque(listOf(result(null), result(4), RuntimeException("bad frame"), result(4))))
        val vm = BoardViewModel({ runtime }, UnconfinedTestDispatcher())
        repeat(4) { vm.onFrame(frame) }
        val state = vm.state.value
        assertIs<BoardUiState.Ready>(state)
        assertEquals(4, state.lastSelection)
        assertEquals(2, state.selectionCount)
        assertEquals(1, state.frameErrors)
        assertEquals(1, state.frame?.selection?.hoverCell)
    }

    @Test
    fun cameraErrorFailsAndClearingClosesRuntime() {
        val runtime = FakeRuntime(info, ArrayDeque())
        val vm = BoardViewModel({ runtime }, UnconfinedTestDispatcher())
        vm.onCameraError(IllegalStateException("no camera"))
        assertEquals(BoardUiState.Failed("Camera: no camera"), vm.state.value)
        val onCleared = BoardViewModel::class.java.getDeclaredMethod("onCleared").apply { isAccessible = true }
        onCleared.invoke(vm)
        assertTrue(runtime.closed)
    }
}
