package app.eyegaze.domain.unit

import app.eyegaze.domain.calibration.CalibrationException
import app.eyegaze.domain.calibration.RidgeCalibrator
import app.eyegaze.domain.geometry.GazeCm
import app.eyegaze.domain.geometry.ScreenGeometry
import app.eyegaze.domain.geometry.ScreenPoint
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import kotlin.random.Random
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class CalibrationUnitTest {
    private fun dataset(
        targets: Int = 9,
        perTarget: Int = 10,
        noise: Double = 0.0,
    ): Triple<List<DoubleArray>, List<DoubleArray>, IntArray> {
        val rnd = Random(1)
        val w = Array(4) { doubleArrayOf(rnd.nextDouble(-1.0, 1.0), rnd.nextDouble(-1.0, 1.0)) }
        val x = mutableListOf<DoubleArray>()
        val y = mutableListOf<DoubleArray>()
        val g = mutableListOf<Int>()
        repeat(targets) { t ->
            val centre = DoubleArray(4) { rnd.nextDouble(-2.0, 2.0) }
            repeat(perTarget) {
                val f = DoubleArray(4) { centre[it] + rnd.nextDouble(-0.05, 0.05) }
                x += f
                y +=
                    DoubleArray(2) { k ->
                        0.5 + (0 until 4).sumOf { f[it] * w[it][k] } +
                            if (noise > 0) rnd.nextDouble(-noise, noise) else 0.0
                    }
                g += t
            }
        }
        return Triple(x, y, g.toIntArray())
    }

    @Test
    fun `recovers a noise free linear map`() {
        val (x, y, g) = dataset()
        val model = RidgeCalibrator.fit(x, y, g, listOf(1e-6, 1.0))
        assertEquals(1e-6, model.lambda)
        x.indices.forEach { i ->
            val p = model.predict(x[i])
            assertEquals(y[i][0], p[0], 1e-3)
            assertEquals(y[i][1], p[1], 1e-3)
        }
        assertTrue(model.cvError < 1e-2)
    }

    @Test
    fun `constant features do not break the fit`() {
        val (x, y, g) = dataset(noise = 0.01)
        val withConstant = x.map { it + 7.0 }
        val model = RidgeCalibrator.fit(withConstant, y, g)
        assertTrue(model.weights.all { row -> row.all { it.isFinite() } })
    }

    @Test
    fun `input validation`() {
        val (x, y, g) = dataset(targets = 2)
        assertThrows<CalibrationException> { RidgeCalibrator.fit(x, y, g) }
        val (x3, y3, g3) = dataset()
        assertThrows<CalibrationException> { RidgeCalibrator.fit(x3.drop(1), y3, g3) }
        assertThrows<CalibrationException> { RidgeCalibrator.fit(x3, y3, g3, listOf(-1.0)) }
        val bad = y3.toMutableList().also { it[0] = doubleArrayOf(Double.NaN, 0.0) }
        assertThrows<CalibrationException> { RidgeCalibrator.fit(x3, bad, g3) }
        val model = RidgeCalibrator.fit(x3, y3, g3)
        assertThrows<IllegalArgumentException> { model.predict(DoubleArray(3)) }
    }

    @Test
    fun `screen geometry roundtrip and distance`() {
        val geo = ScreenGeometry.fromDpi(1080, 2340, 400.0, 400.0)
        val p = ScreenPoint(0.3, 0.8)
        val back = geo.cameraCmToScreen(geo.screenToCameraCm(p))
        assertEquals(p.x, back.x, 1e-12)
        assertEquals(p.y, back.y, 1e-12)
        assertTrue(geo.screenToCameraCm(ScreenPoint(0.5, 0.5)).y < 0)
        assertEquals(geo.widthCm, geo.distanceCm(ScreenPoint(0.0, 0.5), ScreenPoint(1.0, 0.5)), 1e-12)
        assertEquals(0.0, geo.cameraCmToScreen(GazeCm(-geo.cameraXCm, 0.0)).x, 1e-12)
    }
}
