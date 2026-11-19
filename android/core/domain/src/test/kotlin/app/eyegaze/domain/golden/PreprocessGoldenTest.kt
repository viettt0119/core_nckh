package app.eyegaze.domain.golden

import app.eyegaze.domain.face.FaceKeypoints
import app.eyegaze.domain.face.FaceObservation
import app.eyegaze.domain.geometry.Point
import app.eyegaze.domain.preprocess.CropGeometry
import app.eyegaze.domain.preprocess.CropTransform
import app.eyegaze.domain.preprocess.Preprocessor
import app.eyegaze.domain.preprocess.RgbImage
import app.eyegaze.domain.spec.PreprocessSpec
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.jupiter.api.DynamicTest
import org.junit.jupiter.api.TestFactory
import java.io.File
import kotlin.math.abs
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/** Kotlin preprocessing must reproduce the Python reference on every golden case. */
class PreprocessGoldenTest {
    private val spec = PreprocessSpec.parse(File(Golden.contracts, "preprocess_spec.v1.json").readText())
    private val preprocessor = Preprocessor(spec)
    private val scalarTol = Golden.tolerance("scalars")
    private val cropTol = Golden.tolerance("crops")

    private fun assertPoint(
        expected: kotlinx.serialization.json.JsonElement,
        actual: Point,
        what: String,
    ) {
        val e = expected.doubles()
        assertEquals(e[0], actual.x, scalarTol, "$what.x")
        assertEquals(e[1], actual.y, scalarTol, "$what.y")
    }

    private fun assertTransform(
        expected: JsonObject,
        actual: CropTransform,
        what: String,
    ) {
        assertEquals(expected["cx"]!!.d, actual.cx, scalarTol, "$what.cx")
        assertEquals(expected["cy"]!!.d, actual.cy, scalarTol, "$what.cy")
        assertEquals(expected["side"]!!.d, actual.side, scalarTol, "$what.side")
        assertEquals(expected["angle"]!!.d, actual.angle, scalarTol, "$what.angle")
    }

    @TestFactory
    fun cases(): List<DynamicTest> =
        Golden.cases().map { name ->
            DynamicTest.dynamicTest(name) {
                val meta = Golden.json("$name.json")
                val w = meta["frame_width"]!!.i
                val h = meta["frame_height"]!!.i
                val obs = FaceObservation(meta["landmarks"]!!.flatDoubles(), meta["transform_matrix"]!!.flatDoubles(), w, h)
                val expected = meta["expected"]!!.jsonObject

                val kp = FaceKeypoints.from(obs, spec)
                val ek = expected["keypoints"]!!.jsonObject
                assertPoint(ek["right_outer"]!!, kp.rightOuter, "right_outer")
                assertPoint(ek["right_inner"]!!, kp.rightInner, "right_inner")
                assertPoint(ek["left_inner"]!!, kp.leftInner, "left_inner")
                assertPoint(ek["left_outer"]!!, kp.leftOuter, "left_outer")
                assertPoint(ek["right_iris"]!!, kp.rightIris, "right_iris")
                assertPoint(ek["left_iris"]!!, kp.leftIris, "left_iris")
                val bbox = ek["face_bbox"]!!.jsonObject
                assertEquals(bbox["w"]!!.d, kp.faceBox.w, scalarTol, "bbox.w")
                assertEquals(bbox["y"]!!.d, kp.faceBox.y, scalarTol, "bbox.y")
                val pose = ek["pose"]!!.jsonObject
                assertEquals(pose["pitch"]!!.d, kp.pose.pitch, scalarTol, "pitch")
                assertEquals(pose["yaw"]!!.d, kp.pose.yaw, scalarTol, "yaw")
                assertEquals(pose["roll"]!!.d, kp.pose.roll, scalarTol, "roll")

                val t = CropGeometry.cropTransforms(kp, spec)
                val et = expected["transforms"]!!.jsonObject
                assertTransform(et["right_eye"]!!.jsonObject, t.rightEye, "right_eye")
                assertTransform(et["left_eye"]!!.jsonObject, t.leftEye, "left_eye")
                assertTransform(et["face"]!!.jsonObject, t.face, "face")

                val image =
                    RgbImage(w, h, Golden.decodeDeltaX(Golden.gunzip(meta["image"]!!.jsonObject["file"]!!.jsonPrimitive.content), w, h))
                val inputs = preprocessor.process(image, kp)
                val eg = expected["geometry"]!!.doubles()
                assertEquals(eg.size, inputs.geometry.size)
                for (k in eg.indices) assertEquals(eg[k], inputs.geometry[k].toDouble(), 1e-6, "geometry[$k]")

                val crops = expected["crops"]!!.jsonObject
                val flat = Golden.floatsLE(Golden.gunzip(crops["file"]!!.jsonPrimitive.content))
                val actual = listOf(inputs.rightEye, inputs.leftEye, inputs.face)
                var offset = 0
                crops["shapes"]!!.jsonArray.forEachIndexed { idx, shape ->
                    val size = shape.jsonArray.fold(1) { acc, v -> acc * v.i }
                    assertEquals(size, actual[idx].size, "crop $idx size")
                    var maxDiff = 0.0
                    for (k in 0 until size) maxDiff = maxOf(maxDiff, abs(flat[offset + k] - actual[idx][k]).toDouble())
                    assertTrue(maxDiff <= cropTol, "crop $idx max diff $maxDiff > $cropTol")
                    assertTrue(maxDiff <= 1e-5, "crop $idx should match the reference almost exactly, got $maxDiff")
                    offset += size
                }
                assertEquals(flat.size, offset)
            }
        }
}
