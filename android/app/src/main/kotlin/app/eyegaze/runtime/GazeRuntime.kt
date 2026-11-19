package app.eyegaze.runtime

import android.content.Context
import android.os.Build
import android.view.WindowManager
import app.eyegaze.domain.bundle.VerifiedBundle
import app.eyegaze.domain.gaze.GazeEstimator
import app.eyegaze.domain.gaze.UncalibratedMapper
import app.eyegaze.domain.geometry.ScreenGeometry
import app.eyegaze.domain.interaction.CellGrid
import app.eyegaze.domain.interaction.GazeSelectionEngine
import app.eyegaze.domain.pipeline.CameraFrame
import app.eyegaze.domain.pipeline.FramePipeline
import app.eyegaze.domain.pipeline.GazeFrameResult
import app.eyegaze.domain.preprocess.Preprocessor
import app.eyegaze.ml.AssetBundleSource
import app.eyegaze.ml.LiteRtGazeModel
import app.eyegaze.ml.MediaPipeFaceLandmarker
import dagger.hilt.android.qualifiers.ApplicationContext
import java.io.Closeable
import javax.inject.Inject

/** Static facts about the loaded model, shown in the diagnostics overlay. */
data class RuntimeInfo(
    val modelRun: String,
    val modelEpoch: Int,
    val accelerator: String,
    val specSha256: String,
    val grid: CellGrid,
)

/** A running gaze pipeline. [process] is called from the camera thread; [close] from the main thread. */
interface GazeRuntime : Closeable {
    val info: RuntimeInfo

    fun process(frame: CameraFrame): GazeFrameResult
}

fun interface GazeRuntimeFactory {
    /** Builds the runtime; throws [app.eyegaze.domain.bundle.BundleException] when no valid model is installed. */
    fun create(): GazeRuntime
}

/** Physical screen layout of this device, from the window size and the display's real pixel density. */
class ScreenGeometryProvider
    @Inject
    constructor(
        @ApplicationContext private val context: Context,
    ) {
        fun current(): ScreenGeometry {
            val metrics = context.resources.displayMetrics
            val (width, height) =
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                    val bounds = context.getSystemService(WindowManager::class.java).maximumWindowMetrics.bounds
                    bounds.width() to bounds.height()
                } else {
                    metrics.widthPixels to metrics.heightPixels
                }
            return ScreenGeometry.fromDpi(width, height, metrics.xdpi.toDouble(), metrics.ydpi.toDouble())
        }
    }

class DefaultGazeRuntimeFactory
    @Inject
    constructor(
        @ApplicationContext private val context: Context,
        private val screen: ScreenGeometryProvider,
    ) : GazeRuntimeFactory {
        override fun create(): GazeRuntime {
            val bundle = AssetBundleSource(context.assets).load()
            return PipelineRuntime(context, bundle, screen.current(), CellGrid(rows = 3, cols = 2))
        }
    }

private class PipelineRuntime(
    context: Context,
    bundle: VerifiedBundle,
    screen: ScreenGeometry,
    grid: CellGrid,
) : GazeRuntime {
    private val model = LiteRtGazeModel.create(bundle.modelBytes, bundle.spec, bundle.card.embeddingDim)
    private val landmarker = MediaPipeFaceLandmarker(context)
    private val pipeline =
        FramePipeline(
            landmarker = landmarker,
            estimator = GazeEstimator(Preprocessor(bundle.spec), model, UncalibratedMapper(screen)),
            engine = GazeSelectionEngine(grid),
        )
    private val lock = Any()
    private var closed = false

    override val info =
        RuntimeInfo(
            bundle.card.sourceRun,
            bundle.card.sourceEpoch,
            model.accelerator.name,
            bundle.card.specSha256,
            grid,
        )

    override fun process(frame: CameraFrame): GazeFrameResult =
        synchronized(lock) {
            check(!closed) { "Runtime closed" }
            pipeline.process(frame)
        }

    override fun close() =
        synchronized(lock) {
            if (!closed) {
                closed = true
                landmarker.close()
                model.close()
            }
        }
}
