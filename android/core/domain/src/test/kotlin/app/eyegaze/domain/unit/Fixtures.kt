package app.eyegaze.domain.unit

import app.eyegaze.domain.face.HeadPose
import app.eyegaze.domain.preprocess.RgbImage
import app.eyegaze.domain.spec.PreprocessSpec
import java.io.File
import kotlin.math.cos
import kotlin.math.sin
import kotlin.random.Random

object Fixtures {
    val spec: PreprocessSpec by lazy {
        PreprocessSpec.parse(File(System.getProperty("eyegaze.contracts"), "preprocess_spec.v1.json").readText())
    }

    fun randomImage(
        width: Int,
        height: Int,
        seed: Int = 0,
    ): RgbImage {
        val rnd = Random(seed)
        return RgbImage(width, height, ByteArray(width * height * 3) { rnd.nextInt(256).toByte() })
    }

    /** Row-major 4x4 transform with rotation R = Rz(roll) Ry(yaw) Rx(pitch), optionally scaled. */
    fun transform(
        pose: HeadPose,
        scale: Double = 1.0,
    ): DoubleArray {
        val (cp, sp) = cos(pose.pitch) to sin(pose.pitch)
        val (cy, sy) = cos(pose.yaw) to sin(pose.yaw)
        val (cr, sr) = cos(pose.roll) to sin(pose.roll)
        val r =
            arrayOf(
                doubleArrayOf(cr * cy, cr * sy * sp - sr * cp, cr * sy * cp + sr * sp),
                doubleArrayOf(sr * cy, sr * sy * sp + cr * cp, sr * sy * cp - cr * sp),
                doubleArrayOf(-sy, cy * sp, cy * cp),
            )
        val m = DoubleArray(16)
        for (i in 0 until 3) for (j in 0 until 3) m[i * 4 + j] = r[i][j] * scale
        m[15] = 1.0
        return m
    }
}
