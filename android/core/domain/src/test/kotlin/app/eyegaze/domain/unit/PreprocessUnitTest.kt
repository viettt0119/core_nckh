package app.eyegaze.domain.unit

import app.eyegaze.domain.face.HeadPose
import app.eyegaze.domain.preprocess.CropSampler
import app.eyegaze.domain.preprocess.CropTransform
import app.eyegaze.domain.preprocess.RgbImage
import app.eyegaze.domain.spec.PreprocessSpec
import app.eyegaze.domain.spec.SamplingSpec
import app.eyegaze.domain.spec.SpecException
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import java.io.File
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin
import kotlin.test.assertEquals

class PreprocessUnitTest {
    private val oneX = SamplingSpec("bilinear", 1, "replicate", 255.0)

    @Test
    fun `axis aligned unit scale crop reproduces pixels`() {
        val img = Fixtures.randomImage(64, 48)
        val crop = CropSampler.warpCrop(img, CropTransform(10.0 + 8, 7.0 + 8, 16.0, 0.0), 16, oneX)
        for (v in 0 until 16) {
            for (u in 0 until 16) {
                for (c in 0 until 3) {
                    val expected = (img.data[((7 + v) * 64 + 10 + u) * 3 + c].toInt() and 0xFF) / 255.0
                    assertEquals(expected, crop[(v * 16 + u) * 3 + c].toDouble(), 1e-6)
                }
            }
        }
    }

    @Test
    fun `flip mirrors columns`() {
        val img = Fixtures.randomImage(100, 100, 3)
        val t = CropTransform(50.0, 50.0, 30.0, 0.3)
        val plain = CropSampler.warpCrop(img, t, 8, Fixtures.spec.sampling)
        val flipped = CropSampler.warpCrop(img, t, 8, Fixtures.spec.sampling, flipHorizontal = true)
        for (v in 0 until 8) {
            for (u in 0 until 8) {
                for (c in 0 until 3) {
                    assertEquals(plain[(v * 8 + 7 - u) * 3 + c], flipped[(v * 8 + u) * 3 + c])
                }
            }
        }
    }

    @Test
    fun `border is replicated`() {
        val data = ByteArray(20 * 20 * 3)
        for (y in 0 until 20) for (c in 0 until 3) data[(y * 20) * 3 + c] = 200.toByte()
        val crop = CropSampler.warpCrop(RgbImage(20, 20, data), CropTransform(-50.0, 10.0, 4.0, 0.0), 4, oneX)
        crop.forEach { assertEquals(200.0 / 255.0, it.toDouble(), 1e-6) }
    }

    @Test
    fun `bilinear is exact on a linear ramp at any angle`() {
        val size = 100
        val data =
            ByteArray(size * size * 3) { i ->
                val p = i / 3
                ((p % size) + (p / size)).toByte()
            }
        val img = RgbImage(size, size, data)
        for (angle in listOf(0.0, 0.4, -1.2, PI / 3)) {
            val t = CropTransform(48.0, 52.0, 30.0, angle)
            val crop = CropSampler.warpCrop(img, t, 8, oneX)
            for (v in 0 until 8) {
                for (u in 0 until 8) {
                    val a = (u + 0.5) / 8 - 0.5
                    val b = (v + 0.5) / 8 - 0.5
                    val x = t.cx + t.side * (a * cos(angle) - b * sin(angle)) - 0.5
                    val y = t.cy + t.side * (a * sin(angle) + b * cos(angle)) - 0.5
                    assertEquals((x + y) / 255.0, crop[(v * 8 + u) * 3].toDouble(), 1e-5)
                }
            }
        }
    }

    @Test
    fun `head pose roundtrips and ignores uniform scale`() {
        val pose = HeadPose(0.2, -0.4, 0.3)
        val recovered = HeadPose.fromTransform(Fixtures.transform(pose, scale = 2.5))
        assertEquals(pose.pitch, recovered.pitch, 1e-12)
        assertEquals(pose.yaw, recovered.yaw, 1e-12)
        assertEquals(pose.roll, recovered.roll, 1e-12)
        assertThrows<IllegalArgumentException> { HeadPose.fromTransform(DoubleArray(16)) }
    }

    @Test
    fun `spec validation rejects bad input`() {
        val text = File(System.getProperty("eyegaze.contracts"), "preprocess_spec.v1.json").readText()
        assertThrows<SpecException> { PreprocessSpec.parse(text.replace("\"width_scale\": 1.8", "\"width_scale\": 0.5")) }
        assertThrows<SpecException> { PreprocessSpec.parse(text.replace("\"version\": 1", "\"version\": 2")) }
        assertThrows<SpecException> { PreprocessSpec.parse(text.replace("\"dim\": 17", "\"dim\": 17, \"extra\": 1")) }
        assertThrows<SpecException> { PreprocessSpec.parse("{") }
    }

    @Test
    fun `image and transform validation`() {
        assertThrows<IllegalArgumentException> { RgbImage(2, 2, ByteArray(5)) }
        assertThrows<IllegalArgumentException> { CropTransform(0.0, 0.0, 0.0, 0.0) }
        assertThrows<IllegalArgumentException> { CropTransform(Double.NaN, 0.0, 1.0, 0.0) }
    }
}
