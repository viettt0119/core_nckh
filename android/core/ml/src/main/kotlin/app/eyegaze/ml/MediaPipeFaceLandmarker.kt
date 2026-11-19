package app.eyegaze.ml

import android.content.Context
import android.graphics.Bitmap
import app.eyegaze.domain.face.FaceObservation
import app.eyegaze.domain.face.columnMajorToRowMajor
import app.eyegaze.domain.pipeline.CameraFrame
import app.eyegaze.domain.pipeline.FaceDetection
import app.eyegaze.domain.pipeline.FaceLandmarkerPort
import app.eyegaze.domain.spec.LANDMARK_COUNT
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import java.io.Closeable

/**
 * MediaPipe Face Landmarker (the same `face_landmarker.task` used to landmark the training set, ADR-0002).
 *
 * VIDEO mode is used instead of LIVE_STREAM: it is synchronous on the analysis thread (no callback hand-off),
 * still tracks across frames, and the camera's KEEP_ONLY_LATEST strategy already drops frames while busy.
 * Not thread-safe.
 */
public class MediaPipeFaceLandmarker(
    context: Context,
    modelAssetPath: String = DEFAULT_MODEL_ASSET,
    minConfidence: Float = 0.5f,
) : FaceLandmarkerPort,
    Closeable {
    private val landmarker: FaceLandmarker =
        FaceLandmarker.createFromOptions(
            context,
            FaceLandmarker.FaceLandmarkerOptions
                .builder()
                .setBaseOptions(BaseOptions.builder().setModelAssetPath(modelAssetPath).build())
                .setRunningMode(RunningMode.VIDEO)
                .setNumFaces(1)
                .setMinFaceDetectionConfidence(minConfidence)
                .setMinFacePresenceConfidence(minConfidence)
                .setMinTrackingConfidence(minConfidence)
                .setOutputFaceBlendshapes(true)
                .setOutputFacialTransformationMatrixes(true)
                .build(),
        )
    private var bitmap: Bitmap? = null
    private var pixels = IntArray(0)
    private var lastTimestampMs = Long.MIN_VALUE

    override fun detect(frame: CameraFrame): FaceDetection? {
        val image = frame.image
        val bmp = bitmapFor(image.width, image.height)
        val data = image.data
        for (i in pixels.indices) {
            val r = data[i * 3].toInt() and 0xFF
            val g = data[i * 3 + 1].toInt() and 0xFF
            val b = data[i * 3 + 2].toInt() and 0xFF
            pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        bmp.setPixels(pixels, 0, image.width, 0, 0, image.width, image.height)
        // VIDEO mode requires strictly increasing timestamps.
        val timestampMs = maxOf(frame.timestampNanos / 1_000_000, lastTimestampMs + 1)
        lastTimestampMs = timestampMs
        val result = landmarker.detectForVideo(BitmapImageBuilder(bmp).build(), timestampMs)
        return toDetection(result, image.width, image.height)
    }

    override fun close() {
        landmarker.close()
        bitmap?.recycle()
    }

    private fun bitmapFor(
        width: Int,
        height: Int,
    ): Bitmap {
        val current = bitmap
        if (current != null && current.width == width && current.height == height) return current
        current?.recycle()
        pixels = IntArray(width * height)
        return Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).also { bitmap = it }
    }

    public companion object {
        public const val DEFAULT_MODEL_ASSET: String = "face_landmarker.task"
        private const val BLINK_LEFT = "eyeBlinkLeft"
        private const val BLINK_RIGHT = "eyeBlinkRight"

        internal fun toDetection(
            result: FaceLandmarkerResult,
            width: Int,
            height: Int,
        ): FaceDetection? {
            val points = result.faceLandmarks().firstOrNull() ?: return null
            if (points.size != LANDMARK_COUNT) return null
            val matrix = result.facialTransformationMatrixes().orElse(null)?.firstOrNull() ?: return null
            val landmarks = DoubleArray(LANDMARK_COUNT * 3)
            points.forEachIndexed { i, p ->
                landmarks[i * 3] = p.x().toDouble()
                landmarks[i * 3 + 1] = p.y().toDouble()
                landmarks[i * 3 + 2] = p.z().toDouble()
            }
            val scores =
                result
                    .faceBlendshapes()
                    .orElse(null)
                    ?.firstOrNull()
                    ?.associate { it.categoryName() to it.score().toDouble() }
                    .orEmpty()
            val observation =
                try {
                    FaceObservation(landmarks, columnMajorToRowMajor(matrix), width, height)
                } catch (e: IllegalArgumentException) {
                    return null
                }
            return FaceDetection(observation, scores[BLINK_LEFT] ?: 0.0, scores[BLINK_RIGHT] ?: 0.0)
        }
    }
}
