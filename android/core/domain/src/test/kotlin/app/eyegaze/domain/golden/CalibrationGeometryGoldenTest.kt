package app.eyegaze.domain.golden

import app.eyegaze.domain.calibration.RidgeCalibrator
import app.eyegaze.domain.geometry.GazeCm
import app.eyegaze.domain.geometry.ScreenGeometry
import app.eyegaze.domain.spec.PreprocessSpec
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.jupiter.api.Test
import java.io.File
import java.security.MessageDigest
import kotlin.test.assertEquals

class CalibrationGeometryGoldenTest {
    private val tol = Golden.tolerance("scalars")

    @Test
    fun `ridge calibration matches the python reference`() {
        val g = Golden.json("calibration.json")
        val e = g["expected"]!!.jsonObject
        val model =
            RidgeCalibrator.fit(
                features = g["features"]!!.matrix(),
                targets = g["targets"]!!.matrix(),
                groups = g["groups"]!!.jsonArray.map { it.i }.toIntArray(),
                lambdas = g["lambdas"]!!.doubles().toList(),
            )
        assertEquals(e["lambda"]!!.d, model.lambda, 0.0)
        assertEquals(e["cv_error"]!!.d, model.cvError, tol)
        val mean = e["feature_mean"]!!.doubles()
        val std = e["feature_std"]!!.doubles()
        for (j in mean.indices) {
            assertEquals(mean[j], model.featureMean[j], tol)
            assertEquals(std[j], model.featureStd[j], tol)
        }
        e["weights"]!!.matrix().forEachIndexed { r, row -> row.forEachIndexed { c, v -> assertEquals(v, model.weights[r][c], tol) } }
        val probes = e["probes"]!!.matrix()
        val predictions = e["predictions"]!!.matrix()
        probes.forEachIndexed { k, probe ->
            val p = model.predict(probe)
            assertEquals(predictions[k][0], p[0], tol)
            assertEquals(predictions[k][1], p[1], tol)
        }
    }

    @Test
    fun `screen geometry matches the python reference`() {
        val g = Golden.json("screen_geometry.json")
        assertEquals(g["default_camera_y_cm"]!!.d, ScreenGeometry.DEFAULT_CAMERA_Y_CM, 0.0)
        for (case in g["cases"]!!.jsonArray) {
            val c = case.jsonObject
            val s = c["screen"]!!.jsonObject
            val geo = ScreenGeometry.fromDpi(s["width_px"]!!.i, s["height_px"]!!.i, s["xdpi"]!!.d, s["ydpi"]!!.d)
            assertEquals(c["expected_camera_x_cm"]!!.d, geo.cameraXCm, tol)
            assertEquals(c["expected_camera_y_cm"]!!.d, geo.cameraYCm, tol)
            val gaze = c["gaze_cm"]!!.matrix()
            val expected = c["expected_screen_norm"]!!.matrix()
            gaze.forEachIndexed { k, xy ->
                val p = geo.cameraCmToScreen(GazeCm(xy[0], xy[1]))
                assertEquals(expected[k][0], p.x, tol)
                assertEquals(expected[k][1], p.y, tol)
            }
        }
    }

    @Test
    fun `bundled spec parses and matches the fixture hash`() {
        val bytes = File(Golden.contracts, "preprocess_spec.v1.json").readBytes()
        val digest = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
        assertEquals(Golden.json("index.json")["spec_sha256"]!!.jsonPrimitive.content, digest)
        val spec = PreprocessSpec.parse(bytes.decodeToString())
        assertEquals(64, spec.eye.size)
        assertEquals(17, spec.geometry.layout.size)
    }
}
