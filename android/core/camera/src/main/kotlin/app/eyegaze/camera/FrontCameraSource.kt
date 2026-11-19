package app.eyegaze.camera

import android.content.Context
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import app.eyegaze.domain.pipeline.CameraFrame
import app.eyegaze.domain.preprocess.ImageOps
import kotlinx.coroutines.suspendCancellableCoroutine
import java.io.Closeable
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

/**
 * Front camera frames for the gaze pipeline: RGBA analysis stream (~640x480), converted to upright,
 * un-mirrored RGB on a dedicated thread. KEEP_ONLY_LATEST drops frames while the pipeline is busy, which
 * bounds latency instead of letting a queue build up.
 */
public class FrontCameraSource(
    private val context: Context,
    private val targetSize: Size = Size(640, 480),
) {
    /** Binds the camera to [owner]; [onFrame] runs on the analysis thread. Close the handle to unbind. */
    public suspend fun bind(
        owner: LifecycleOwner,
        onFrame: (CameraFrame) -> Unit,
        onError: (Throwable) -> Unit,
    ): Closeable {
        val provider = awaitProvider()
        if (!provider.hasCamera(CameraSelector.DEFAULT_FRONT_CAMERA)) {
            throw IllegalStateException("This device has no front camera")
        }
        val executor: ExecutorService = Executors.newSingleThreadExecutor { r -> Thread(r, "gaze-analysis") }
        val analysis =
            ImageAnalysis
                .Builder()
                .setResolutionSelector(
                    ResolutionSelector
                        .Builder()
                        .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                        .setResolutionStrategy(
                            ResolutionStrategy(targetSize, ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER),
                        ).build(),
                ).setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
        var scratch = ByteArray(0)
        analysis.setAnalyzer(executor) { image: ImageProxy ->
            try {
                val plane = image.planes[0]
                val buffer = plane.buffer
                if (scratch.size != buffer.remaining()) scratch = ByteArray(buffer.remaining())
                buffer.get(scratch)
                val rgb =
                    ImageOps.rgbaToUprightRgb(
                        scratch,
                        image.width,
                        image.height,
                        plane.rowStride,
                        image.imageInfo.rotationDegrees,
                    )
                onFrame(CameraFrame(rgb, image.imageInfo.timestamp))
            } catch (e: Exception) {
                onError(e)
            } finally {
                image.close()
            }
        }
        provider.unbindAll()
        provider.bindToLifecycle(owner, CameraSelector.DEFAULT_FRONT_CAMERA, analysis)
        return Closeable {
            analysis.clearAnalyzer()
            provider.unbind(analysis)
            executor.shutdown()
        }
    }

    private suspend fun awaitProvider(): ProcessCameraProvider =
        suspendCancellableCoroutine { cont ->
            val future = ProcessCameraProvider.getInstance(context)
            future.addListener(
                {
                    try {
                        cont.resume(future.get())
                    } catch (e: Exception) {
                        cont.resumeWithException(e)
                    }
                },
                ContextCompat.getMainExecutor(context),
            )
        }
}
