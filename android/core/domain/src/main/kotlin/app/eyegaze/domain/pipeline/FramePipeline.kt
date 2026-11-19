package app.eyegaze.domain.pipeline

import app.eyegaze.domain.face.FaceObservation
import app.eyegaze.domain.face.HeadPose
import app.eyegaze.domain.gaze.GazeEstimator
import app.eyegaze.domain.geometry.ScreenPoint
import app.eyegaze.domain.interaction.FrameSignal
import app.eyegaze.domain.interaction.GazeSelectionEngine
import app.eyegaze.domain.interaction.SelectionState
import app.eyegaze.domain.preprocess.RgbImage

/** One upright, un-mirrored camera frame. */
public class CameraFrame(
    public val image: RgbImage,
    public val timestampNanos: Long,
)

/** Landmarker output for one frame. */
public class FaceDetection(
    public val observation: FaceObservation,
    public val blinkLeft: Double,
    public val blinkRight: Double,
)

/** Port implemented by the MediaPipe adapter (`:core:ml`). */
public fun interface FaceLandmarkerPort {
    public fun detect(frame: CameraFrame): FaceDetection?
}

/** Exponential moving average of a duration in milliseconds. */
public class LatencyTracker(
    private val smoothing: Double = 0.1,
) {
    init {
        require(smoothing in 0.0..1.0 && smoothing > 0.0) { "smoothing must be in (0, 1]" }
    }

    public var averageMs: Double = 0.0
        private set
    private var initialized = false

    public fun record(nanos: Long) {
        val ms = nanos / 1e6
        averageMs = if (initialized) averageMs + smoothing * (ms - averageMs) else ms
        initialized = true
    }
}

public data class PipelineMetrics(
    val fps: Double,
    val landmarkMs: Double,
    val gazeMs: Double,
    val totalMs: Double,
)

public data class GazeFrameResult(
    val gaze: ScreenPoint?,
    val rawGaze: ScreenPoint?,
    val pose: HeadPose?,
    val selection: SelectionState,
    val metrics: PipelineMetrics,
)

/**
 * Per-frame orchestration: landmarks → gaze estimate → look-to-select / blink-to-confirm.
 * Runs on the camera analysis thread; frames that arrive while it is busy are dropped upstream
 * (CameraX KEEP_ONLY_LATEST), which keeps latency bounded instead of queueing.
 */
public class FramePipeline(
    private val landmarker: FaceLandmarkerPort,
    private val estimator: GazeEstimator,
    private val engine: GazeSelectionEngine,
    private val clockNanos: () -> Long = System::nanoTime,
) {
    private val landmarkLatency = LatencyTracker()
    private val gazeLatency = LatencyTracker()
    private val totalLatency = LatencyTracker()
    private var lastFrameNanos: Long? = null
    private var fps = 0.0

    public fun process(frame: CameraFrame): GazeFrameResult {
        val start = clockNanos()
        lastFrameNanos?.let { previous ->
            val dt = (frame.timestampNanos - previous) / 1e9
            if (dt > 0) fps = if (fps == 0.0) 1.0 / dt else fps + 0.1 * (1.0 / dt - fps)
        }
        lastFrameNanos = frame.timestampNanos
        val timeMs = frame.timestampNanos / 1_000_000

        val detection = landmarker.detect(frame)
        val afterLandmarks = clockNanos()
        landmarkLatency.record(afterLandmarks - start)

        if (detection == null) {
            estimator.reset()
            val selection = engine.onFrame(FrameSignal(timeMs, null, 0.0, 0.0))
            totalLatency.record(clockNanos() - start)
            return GazeFrameResult(null, null, null, selection, metrics())
        }
        val estimate = estimator.estimate(frame.image, detection.observation, frame.timestampNanos)
        val afterGaze = clockNanos()
        gazeLatency.record(afterGaze - afterLandmarks)
        val selection =
            engine.onFrame(FrameSignal(timeMs, estimate.smoothedScreen, detection.blinkLeft, detection.blinkRight))
        totalLatency.record(clockNanos() - start)
        return GazeFrameResult(
            estimate.smoothedScreen,
            estimate.rawScreen,
            estimate.keypoints.pose,
            selection,
            metrics(),
        )
    }

    private fun metrics() =
        PipelineMetrics(fps, landmarkLatency.averageMs, gazeLatency.averageMs, totalLatency.averageMs)
}
