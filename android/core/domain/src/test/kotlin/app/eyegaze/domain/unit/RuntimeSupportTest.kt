package app.eyegaze.domain.unit

import app.eyegaze.domain.bundle.BundleException
import app.eyegaze.domain.bundle.BundleVerifier
import app.eyegaze.domain.face.FaceObservation
import app.eyegaze.domain.face.HeadPose
import app.eyegaze.domain.face.columnMajorToRowMajor
import app.eyegaze.domain.gaze.GazeEstimator
import app.eyegaze.domain.gaze.GazeModel
import app.eyegaze.domain.gaze.GazePrediction
import app.eyegaze.domain.gaze.UncalibratedMapper
import app.eyegaze.domain.geometry.GazeCm
import app.eyegaze.domain.geometry.ScreenGeometry
import app.eyegaze.domain.interaction.CellGrid
import app.eyegaze.domain.interaction.GazeSelectionEngine
import app.eyegaze.domain.pipeline.CameraFrame
import app.eyegaze.domain.pipeline.FaceDetection
import app.eyegaze.domain.pipeline.FramePipeline
import app.eyegaze.domain.preprocess.ImageOps
import app.eyegaze.domain.preprocess.ModelInputs
import app.eyegaze.domain.preprocess.Preprocessor
import app.eyegaze.domain.spec.LANDMARK_COUNT
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import java.io.File
import kotlin.random.Random
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class TransformMatrixTest {
    @Test
    fun `column major packing converts back to the row major matrix python produces`() {
        val rowMajor =
            Fixtures.transform(HeadPose(0.2, -0.1, 0.3), scale = 1.1).also {
                it[3] = 1.5
                it[7] = -2.0
                it[11] = -40.0
            }
        val packed = FloatArray(16) { i -> rowMajor[(i % 4) * 4 + i / 4].toFloat() } // what MediaPipe Java returns
        val converted = columnMajorToRowMajor(packed)
        for (i in 0 until 16) assertEquals(rowMajor[i], converted[i], 1e-6)
        val pose = HeadPose.fromTransform(converted)
        assertEquals(0.3, pose.roll, 1e-6)
        assertThrows<IllegalArgumentException> { columnMajorToRowMajor(FloatArray(9)) }
    }
}

class ImageOpsTest {
    // 3x2 RGBA image; pixel value encodes (x, y) in the red channel.
    private fun source(stride: Int = 12): ByteArray {
        val bytes = ByteArray(stride * 2)
        for (y in 0 until 2) {
            for (x in 0 until 3) {
                val i = y * stride + x * 4
                bytes[i] = (10 * y + x).toByte()
                bytes[i + 1] = 100
                bytes[i + 2] = 50
                bytes[i + 3] = -1
            }
        }
        return bytes
    }

    private fun red(img: app.eyegaze.domain.preprocess.RgbImage) = IntArray(img.width * img.height) { img.data[it * 3].toInt() }

    @Test
    fun `rotations are clockwise and drop alpha`() {
        val r0 = ImageOps.rgbaToUprightRgb(source(), 3, 2, 12, 0)
        assertContentEquals(intArrayOf(0, 1, 2, 10, 11, 12), red(r0))
        assertEquals(100, r0.data[1].toInt())
        val r90 = ImageOps.rgbaToUprightRgb(source(), 3, 2, 12, 90)
        assertEquals(2, r90.width)
        assertContentEquals(intArrayOf(10, 0, 11, 1, 12, 2), red(r90))
        val r180 = ImageOps.rgbaToUprightRgb(source(), 3, 2, 12, 180)
        assertContentEquals(intArrayOf(12, 11, 10, 2, 1, 0), red(r180))
        val r270 = ImageOps.rgbaToUprightRgb(source(), 3, 2, 12, 270)
        assertContentEquals(intArrayOf(2, 12, 1, 11, 0, 10), red(r270))
    }

    @Test
    fun `row padding is skipped and bad input rejected`() {
        val padded = ImageOps.rgbaToUprightRgb(source(stride = 16), 3, 2, 16, 0)
        assertContentEquals(intArrayOf(0, 1, 2, 10, 11, 12), red(padded))
        assertThrows<IllegalArgumentException> { ImageOps.rgbaToUprightRgb(source(), 3, 2, 8, 0) }
        assertThrows<IllegalArgumentException> { ImageOps.rgbaToUprightRgb(source(), 3, 2, 12, 45) }
    }
}

class BundleVerifierTest {
    private val spec = File(System.getProperty("eyegaze.contracts"), "preprocess_spec.v1.json").readBytes()
    private val model = byteArrayOf(1, 2, 3, 4)

    private fun card(
        modelSha: String = BundleVerifier.sha256(model),
        format: String = "eyegaze-model-card-v1",
        io: Int = 1,
    ) = """
        {"format":"$format","model_file":"gaze_model.tflite","model_sha256":"$modelSha","spec_file":"preprocess_spec.json",
        "spec_sha256":"${BundleVerifier.sha256(spec)}","model_io_version":$io,"embedding_dim":64,"precision":"fp32",
        "source_run":"base","source_epoch":7,"future_field":{"x":1}}
        """.trimIndent()

    private fun files(cardText: String?) =
        mapOf("model_card.json" to cardText?.encodeToByteArray(), "gaze_model.tflite" to model, "preprocess_spec.json" to spec)

    @Test
    fun `valid bundle verifies`() {
        val bundle = BundleVerifier.verify { files(card())[it] }
        assertEquals("base", bundle.card.sourceRun)
        assertEquals(64, bundle.spec.eye.size)
        assertContentEquals(model, bundle.modelBytes)
    }

    @Test
    fun `missing, tampered or incompatible bundles are refused`() {
        assertThrows<BundleException> { BundleVerifier.verify { files(null)[it] } }
        assertThrows<BundleException> { BundleVerifier.verify { files(card(modelSha = "0".repeat(64)))[it] } }
        assertThrows<BundleException> { BundleVerifier.verify { files(card(format = "other"))[it] } }
        assertThrows<BundleException> { BundleVerifier.verify { files(card(io = 2))[it] } }
        assertThrows<BundleException> { BundleVerifier.verify { files("{")[it] } }
        assertThrows<BundleException> { BundleVerifier.verify { name -> if (name == "gaze_model.tflite") null else files(card())[name] } }
    }
}

class FramePipelineTest {
    private class FakeModel : GazeModel {
        override val embeddingDim = 2

        override fun predict(inputs: ModelInputs) = GazePrediction(GazeCm(0.0, -6.0), floatArrayOf(0f, 0f))
    }

    private fun observation(): FaceObservation {
        val rnd = Random(5)
        val lm = DoubleArray(LANDMARK_COUNT * 3) { if (it % 3 == 2) 0.0 else rnd.nextDouble(0.35, 0.65) }

        fun set(
            idx: Int,
            x: Double,
            y: Double,
        ) {
            lm[idx * 3] = x
            lm[idx * 3 + 1] = y
        }
        set(33, 0.38, 0.45)
        set(133, 0.46, 0.45)
        set(362, 0.54, 0.45)
        set(263, 0.62, 0.45)
        return FaceObservation(lm, Fixtures.transform(HeadPose(0.0, 0.0, 0.0)), 120, 160)
    }

    @Test
    fun `face and no face frames produce metrics and selection state`() {
        var now = 0L
        var faceVisible = true
        val obs = observation()
        val pipeline =
            FramePipeline(
                landmarker = { if (faceVisible) FaceDetection(obs, 0.05, 0.05) else null },
                estimator =
                    GazeEstimator(
                        Preprocessor(Fixtures.spec),
                        FakeModel(),
                        UncalibratedMapper(ScreenGeometry.fromDpi(1080, 2340, 400.0, 400.0)),
                    ),
                engine = GazeSelectionEngine(CellGrid(2, 2)),
                clockNanos = { now.also { now += 1_000_000 } },
            )
        val image = Fixtures.randomImage(120, 160)
        var last = pipeline.process(CameraFrame(image, 0))
        for (k in 1..20) last = pipeline.process(CameraFrame(image, k * 33_333_333L))
        assertNotNull(last.gaze)
        assertTrue(last.selection.faceVisible)
        assertEquals(30.0, last.metrics.fps, 0.5)
        assertTrue(last.metrics.totalMs > 0 && last.metrics.landmarkMs > 0 && last.metrics.gazeMs > 0)
        assertNotNull(last.selection.hoverCell)

        faceVisible = false
        val lost = pipeline.process(CameraFrame(image, 21 * 33_333_333L))
        assertNull(lost.gaze)
        assertEquals(false, lost.selection.faceVisible)
    }
}
