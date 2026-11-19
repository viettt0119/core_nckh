package app.eyegaze.domain.face

import app.eyegaze.domain.geometry.Box
import app.eyegaze.domain.geometry.Point
import app.eyegaze.domain.spec.LANDMARK_COUNT
import app.eyegaze.domain.spec.PreprocessSpec
import kotlin.math.asin
import kotlin.math.atan2
import kotlin.math.sqrt

/**
 * One face from the landmarker, in the units the preprocess spec prescribes.
 *
 * @property landmarks 478×3 values, row-major (x, y, z); x/y normalized by frame width/height.
 * @property transform 4×4 facial transformation matrix, **row-major** (convert from MediaPipe's
 *   column-major Android output at the adapter boundary).
 */
public class FaceObservation(
    public val landmarks: DoubleArray,
    public val transform: DoubleArray,
    public val frameWidth: Int,
    public val frameHeight: Int,
) {
    init {
        require(landmarks.size == LANDMARK_COUNT * 3) {
            "Expected ${LANDMARK_COUNT * 3} landmark values, got ${landmarks.size}"
        }
        require(transform.size == 16) { "Expected 16 transform values, got ${transform.size}" }
        require(frameWidth > 0 && frameHeight > 0) { "Frame size must be positive" }
        require(landmarks.all { it.isFinite() } && transform.all { it.isFinite() }) {
            "Landmarks and transform must be finite"
        }
    }

    public fun px(index: Int): Point = Point(landmarks[index * 3] * frameWidth, landmarks[index * 3 + 1] * frameHeight)

    public fun headPose(): HeadPose = HeadPose.fromTransform(transform)
}

/** Euler angles in radians with R = Rz(roll) · Ry(yaw) · Rx(pitch). */
public data class HeadPose(
    val pitch: Double,
    val yaw: Double,
    val roll: Double,
) {
    public companion object {
        /** Extracts the pose from a row-major 4×4 rigid transform that may carry a uniform scale. */
        public fun fromTransform(m: DoubleArray): HeadPose {
            require(m.size == 16) { "Expected a 4x4 matrix" }

            fun at(
                r: Int,
                c: Int,
            ) = m[r * 4 + c]
            val scales = DoubleArray(3) { c -> sqrt(at(0, c) * at(0, c) + at(1, c) * at(1, c) + at(2, c) * at(2, c)) }
            require(scales.all { it >= 1e-9 }) { "Degenerate transform: zero-length basis vector" }

            fun r(
                row: Int,
                col: Int,
            ) = at(row, col) / scales[col]
            return HeadPose(
                pitch = atan2(r(2, 1), r(2, 2)),
                yaw = asin((-r(2, 0)).coerceIn(-1.0, 1.0)),
                roll = atan2(r(1, 0), r(0, 0)),
            )
        }
    }
}

/** The six spec points, face bounding box and head pose — all the preprocessing needs. */
public data class FaceKeypoints(
    val frameWidth: Int,
    val frameHeight: Int,
    val rightOuter: Point,
    val rightInner: Point,
    val leftInner: Point,
    val leftOuter: Point,
    val rightIris: Point,
    val leftIris: Point,
    val faceBox: Box,
    val pose: HeadPose,
) {
    public fun points(): List<Point> = listOf(rightOuter, rightInner, leftInner, leftOuter, rightIris, leftIris)

    public companion object {
        public fun from(
            obs: FaceObservation,
            spec: PreprocessSpec,
        ): FaceKeypoints {
            val e = spec.eye
            var minX = Double.POSITIVE_INFINITY
            var minY = Double.POSITIVE_INFINITY
            var maxX = Double.NEGATIVE_INFINITY
            var maxY = Double.NEGATIVE_INFINITY
            for (i in 0 until spec.face.bboxLandmarkCount) {
                val x = obs.landmarks[i * 3] * obs.frameWidth
                val y = obs.landmarks[i * 3 + 1] * obs.frameHeight
                if (x < minX) minX = x
                if (x > maxX) maxX = x
                if (y < minY) minY = y
                if (y > maxY) maxY = y
            }
            val box = Box(minX, minY, maxOf(maxX - minX, 1e-6), maxOf(maxY - minY, 1e-6))
            return FaceKeypoints(
                frameWidth = obs.frameWidth,
                frameHeight = obs.frameHeight,
                rightOuter = obs.px(e.right.outer),
                rightInner = obs.px(e.right.inner),
                leftInner = obs.px(e.left.inner),
                leftOuter = obs.px(e.left.outer),
                rightIris = obs.px(e.right.irisCenter),
                leftIris = obs.px(e.left.irisCenter),
                faceBox = box,
                pose = obs.headPose(),
            )
        }
    }
}
