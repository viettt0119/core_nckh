package app.eyegaze.domain.unit

import app.eyegaze.domain.calibration.RidgeCalibrator
import app.eyegaze.domain.face.FaceObservation
import app.eyegaze.domain.face.HeadPose
import app.eyegaze.domain.gaze.CalibratedMapper
import app.eyegaze.domain.gaze.GazeEstimator
import app.eyegaze.domain.gaze.GazeModel
import app.eyegaze.domain.gaze.GazePrediction
import app.eyegaze.domain.gaze.UncalibratedMapper
import app.eyegaze.domain.gaze.calibrationFeatures
import app.eyegaze.domain.geometry.GazeCm
import app.eyegaze.domain.geometry.ScreenGeometry
import app.eyegaze.domain.preprocess.ModelInputs
import app.eyegaze.domain.preprocess.Preprocessor
import app.eyegaze.domain.spec.LANDMARK_COUNT
import org.junit.jupiter.api.Test
import kotlin.random.Random
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

private class FixedModel(
    private val gaze: GazeCm,
) : GazeModel {
    var lastInputs: ModelInputs? = null
    override val embeddingDim: Int = 4

    override fun predict(inputs: ModelInputs): GazePrediction {
        lastInputs = inputs
        return GazePrediction(gaze, floatArrayOf(1f, 2f, 3f, 4f))
    }
}

class GazeEstimatorTest {
    private val spec = Fixtures.spec

    private fun observation(): FaceObservation {
        val rnd = Random(3)
        val lm = DoubleArray(LANDMARK_COUNT * 3)
        for (i in 0 until LANDMARK_COUNT) {
            lm[i * 3] = rnd.nextDouble(0.35, 0.65)
            lm[i * 3 + 1] = rnd.nextDouble(0.35, 0.65)
        }

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
        set(468, 0.42, 0.45)
        set(473, 0.58, 0.45)
        return FaceObservation(lm, Fixtures.transform(HeadPose(0.1, 0.0, 0.0)), 480, 640)
    }

    @Test
    fun `uncalibrated pipeline feeds spec shaped inputs and maps via screen geometry`() {
        val screen = ScreenGeometry.fromDpi(1080, 2340, 400.0, 400.0)
        val model = FixedModel(GazeCm(0.0, -7.0))
        val estimator = GazeEstimator(Preprocessor(spec), model, UncalibratedMapper(screen))
        val estimate = estimator.estimate(Fixtures.randomImage(480, 640), observation(), 0L)
        val inputs = model.lastInputs!!
        assertEquals(spec.eye.size * spec.eye.size * 3, inputs.rightEye.size)
        assertEquals(spec.face.size * spec.face.size * 3, inputs.face.size)
        assertEquals(17, inputs.geometry.size)
        val expected = screen.cameraCmToScreen(GazeCm(0.0, -7.0))
        assertEquals(expected.x, estimate.rawScreen.x, 1e-12)
        assertEquals(expected.y, estimate.smoothedScreen.y, 1e-12) // first sample passes through the filter
    }

    @Test
    fun `calibration features layout and calibrated mapping`() {
        val pose = HeadPose(0.1, 0.2, 0.3)
        val prediction = GazePrediction(GazeCm(1.5, -4.0), floatArrayOf(9f, 8f))
        assertContentEquals(doubleArrayOf(9.0, 8.0, 1.5, -4.0, 0.1, 0.2, 0.3), calibrationFeatures(prediction, pose))

        // Identity-like calibration: screen = (gaze_x / 10 + 0.5, -gaze_y / 20) learned from 9 targets.
        val features = mutableListOf<DoubleArray>()
        val targets = mutableListOf<DoubleArray>()
        val groups = mutableListOf<Int>()
        val rnd = Random(0)
        for (g in 0 until 9) {
            val gx = rnd.nextDouble(-4.0, 4.0)
            val gy = rnd.nextDouble(-12.0, -1.0)
            repeat(6) {
                val p = GazePrediction(GazeCm(gx + rnd.nextDouble(-0.01, 0.01), gy), floatArrayOf(rnd.nextFloat(), rnd.nextFloat()))
                features += calibrationFeatures(p, pose)
                targets += doubleArrayOf(gx / 10 + 0.5, -gy / 20)
                groups += g
            }
        }
        val mapper = CalibratedMapper(RidgeCalibrator.fit(features, targets, groups.toIntArray()))
        val mapped = mapper.map(GazePrediction(GazeCm(2.0, -6.0), floatArrayOf(0.5f, 0.5f)), pose)
        assertEquals(0.7, mapped.x, 0.02)
        assertEquals(0.3, mapped.y, 0.02)
    }
}
