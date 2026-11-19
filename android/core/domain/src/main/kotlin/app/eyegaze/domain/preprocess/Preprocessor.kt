package app.eyegaze.domain.preprocess

import app.eyegaze.domain.face.FaceKeypoints
import app.eyegaze.domain.face.FaceObservation
import app.eyegaze.domain.geometry.Point
import app.eyegaze.domain.spec.EyeSpec
import app.eyegaze.domain.spec.FaceSpec
import app.eyegaze.domain.spec.PreprocessSpec
import app.eyegaze.domain.spec.SamplingSpec
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.sin

// Port of the Python reference preprocessing (research/src/eyegaze/domain/preprocess.py).
// Every arithmetic step follows the reference exactly; contracts/golden fixtures enforce agreement.

/** Upright, un-mirrored RGB frame, 8 bits per channel, interleaved, row-major. */
public class RgbImage(
    public val width: Int,
    public val height: Int,
    public val data: ByteArray,
) {
    init {
        require(width > 0 && height > 0) { "Image size must be positive" }
        require(data.size == width * height * 3) { "Expected ${width * height * 3} bytes, got ${data.size}" }
    }

    internal fun channel(
        x: Int,
        y: Int,
        c: Int,
    ): Int = data[(y * width + x) * 3 + c].toInt() and 0xFF
}

/** Rotated square crop: centre and side in pixels, rotation in radians. */
public data class CropTransform(
    val cx: Double,
    val cy: Double,
    val side: Double,
    val angle: Double,
) {
    init {
        require(cx.isFinite() && cy.isFinite() && angle.isFinite()) { "CropTransform values must be finite" }
        require(side.isFinite() && side > 0) { "CropTransform.side must be positive" }
    }
}

public data class CropTransforms(
    val rightEye: CropTransform,
    val leftEye: CropTransform,
    val face: CropTransform,
)

/** Network inputs for one frame, HWC float32 in [0, 1] (no batch axis), plus the geometry vector. */
public class ModelInputs(
    public val rightEye: FloatArray,
    public val leftEye: FloatArray,
    public val face: FloatArray,
    public val geometry: FloatArray,
)

public object CropSampler {
    /** Extracts a rotated square crop as HWC float32 (size × size × 3); see the reference for the exact definition. */
    public fun warpCrop(
        image: RgbImage,
        t: CropTransform,
        outSize: Int,
        sampling: SamplingSpec,
        flipHorizontal: Boolean = false,
    ): FloatArray {
        require(outSize > 0) { "outSize must be positive" }
        val s = sampling.supersample
        val n = outSize * s
        val grid = DoubleArray(n) { k -> (k / s + ((k % s) + 0.5) / s) / outSize - 0.5 }
        val cosT = cos(t.angle)
        val sinT = sin(t.angle)
        val acc = DoubleArray(outSize * outSize * 3)
        val maxX = image.width - 1
        val maxY = image.height - 1
        for (row in 0 until n) {
            val b = grid[row]
            val v = row / s
            for (col in 0 until n) {
                val a = grid[col]
                val x = t.cx + t.side * (a * cosT - b * sinT) - 0.5
                val y = t.cy + t.side * (a * sinT + b * cosT) - 0.5
                val x0f = floor(x)
                val y0f = floor(y)
                val fx = x - x0f
                val fy = y - y0f
                val x0 = x0f.toInt()
                val y0 = y0f.toInt()
                val x1 = (x0 + 1).coerceIn(0, maxX)
                val y1 = (y0 + 1).coerceIn(0, maxY)
                val x0c = x0.coerceIn(0, maxX)
                val y0c = y0.coerceIn(0, maxY)
                val u = col / s
                val base = (v * outSize + u) * 3
                for (c in 0 until 3) {
                    val top = image.channel(x0c, y0c, c) * (1.0 - fx) + image.channel(x1, y0c, c) * fx
                    val bottom = image.channel(x0c, y1, c) * (1.0 - fx) + image.channel(x1, y1, c) * fx
                    acc[base + c] += top * (1.0 - fy) + bottom * fy
                }
            }
        }
        val norm = (s * s).toDouble() * sampling.pixelDivisor
        val out = FloatArray(outSize * outSize * 3)
        for (v in 0 until outSize) {
            for (u in 0 until outSize) {
                val srcU = if (flipHorizontal) outSize - 1 - u else u
                for (c in 0 until 3) {
                    out[(v * outSize + u) * 3 + c] = (acc[(v * outSize + srcU) * 3 + c] / norm).toFloat()
                }
            }
        }
        return out
    }
}

public object CropGeometry {
    private fun eyeTransform(
        left: Point,
        right: Point,
        spec: EyeSpec,
    ): CropTransform {
        val dx = right.x - left.x
        val dy = right.y - left.y
        val width = hypot(dx, dy)
        require(width > 1e-6) { "Degenerate eye: corners coincide" }
        return CropTransform(
            cx = (left.x + right.x) / 2.0,
            cy = (left.y + right.y) / 2.0,
            side = width * spec.widthScale,
            angle = if (spec.alignRoll) atan2(dy, dx) else 0.0,
        )
    }

    /** Subject's right eye sits on the image left in an un-mirrored frame: its outer corner is the left point. */
    public fun eyeTransforms(
        kp: FaceKeypoints,
        spec: EyeSpec,
    ): Pair<CropTransform, CropTransform> =
        eyeTransform(kp.rightOuter, kp.rightInner, spec) to eyeTransform(kp.leftInner, kp.leftOuter, spec)

    public fun faceTransform(
        kp: FaceKeypoints,
        spec: FaceSpec,
    ): CropTransform {
        val box = kp.faceBox
        val angle =
            if (spec.alignRoll) {
                atan2(
                    kp.leftOuter.y - kp.rightOuter.y,
                    kp.leftOuter.x - kp.rightOuter.x,
                )
            } else {
                0.0
            }
        return CropTransform(box.cx, box.cy, maxOf(box.w, box.h) * spec.scale, angle)
    }

    public fun cropTransforms(
        kp: FaceKeypoints,
        spec: PreprocessSpec,
    ): CropTransforms {
        val (right, left) = eyeTransforms(kp, spec.eye)
        return CropTransforms(right, left, faceTransform(kp, spec.face))
    }
}

public object GeometryFeatures {
    public const val DIM: Int = 17

    /** The 17-D vector in `spec.geometry.layout` order: points (normalized), pitch/yaw/roll, face bbox size. */
    public fun vector(
        kp: FaceKeypoints,
        spec: PreprocessSpec,
    ): FloatArray {
        require(
            spec.geometry.dim == DIM,
        ) { "This implementation produces $DIM features, spec asks ${spec.geometry.dim}" }
        val w = kp.frameWidth.toDouble()
        val h = kp.frameHeight.toDouble()
        val out = FloatArray(DIM)
        var i = 0
        for (p in kp.points()) {
            out[i++] = (p.x / w).toFloat()
            out[i++] = (p.y / h).toFloat()
        }
        out[i++] = kp.pose.pitch.toFloat()
        out[i++] = kp.pose.yaw.toFloat()
        out[i++] = kp.pose.roll.toFloat()
        out[i++] = (kp.faceBox.w / w).toFloat()
        out[i] = (kp.faceBox.h / h).toFloat()
        return out
    }
}

/** Full per-frame preprocessing for one spec: keypoints → crop transforms → crops + geometry. */
public class Preprocessor(
    public val spec: PreprocessSpec,
) {
    public fun keypoints(obs: FaceObservation): FaceKeypoints = FaceKeypoints.from(obs, spec)

    public fun process(
        image: RgbImage,
        obs: FaceObservation,
    ): ModelInputs = process(image, keypoints(obs))

    public fun process(
        image: RgbImage,
        kp: FaceKeypoints,
    ): ModelInputs {
        require(image.width == kp.frameWidth && image.height == kp.frameHeight) {
            "Image ${image.width}x${image.height} does not match keypoints ${kp.frameWidth}x${kp.frameHeight}"
        }
        val t = CropGeometry.cropTransforms(kp, spec)
        return ModelInputs(
            rightEye = CropSampler.warpCrop(image, t.rightEye, spec.eye.size, spec.sampling),
            leftEye = CropSampler.warpCrop(image, t.leftEye, spec.eye.size, spec.sampling, spec.eye.flipLeft),
            face = CropSampler.warpCrop(image, t.face, spec.face.size, spec.sampling),
            geometry = GeometryFeatures.vector(kp, spec),
        )
    }
}
