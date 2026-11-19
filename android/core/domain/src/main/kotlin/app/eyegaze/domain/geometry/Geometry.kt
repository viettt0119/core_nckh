package app.eyegaze.domain.geometry

/** Axis-aligned box in pixels; (x, y) is the top-left corner. */
public data class Box(
    val x: Double,
    val y: Double,
    val w: Double,
    val h: Double,
) {
    init {
        require(x.isFinite() && y.isFinite() && w.isFinite() && h.isFinite()) { "Box values must be finite" }
        require(w > 0 && h > 0) { "Box must have positive size, got w=$w h=$h" }
    }

    val cx: Double get() = x + w / 2.0
    val cy: Double get() = y + h / 2.0
}

public data class Point(
    val x: Double,
    val y: Double,
)

/** A point in normalized screen space: (0, 0) top-left, (1, 1) bottom-right. */
public data class ScreenPoint(
    val x: Double,
    val y: Double,
)

/** Gaze in camera space (cm): origin at the front camera, +x right, +y up (GazeCapture XCam/YCam). */
public data class GazeCm(
    val x: Double,
    val y: Double,
)

/**
 * Physical display layout relative to the front camera. Mirrors `eyegaze.domain.geometry.ScreenGeometry`.
 * `cameraXCm`/`cameraYCm` locate the camera in screen coordinates (cm from the top-left corner, +y down).
 */
public data class ScreenGeometry(
    val widthPx: Int,
    val heightPx: Int,
    val pxPerCmX: Double,
    val pxPerCmY: Double,
    val cameraXCm: Double,
    val cameraYCm: Double,
) {
    init {
        require(widthPx > 0 && heightPx > 0) { "Screen size must be positive" }
        require(pxPerCmX > 0 && pxPerCmY > 0) { "Pixel density must be positive" }
    }

    val widthCm: Double get() = widthPx / pxPerCmX
    val heightCm: Double get() = heightPx / pxPerCmY

    public fun cameraCmToScreen(gaze: GazeCm): ScreenPoint =
        ScreenPoint((cameraXCm + gaze.x) / widthCm, (cameraYCm - gaze.y) / heightCm)

    public fun screenToCameraCm(point: ScreenPoint): GazeCm =
        GazeCm(point.x * widthCm - cameraXCm, cameraYCm - point.y * heightCm)

    /** Euclidean distance in cm between two normalized screen points (for error reporting). */
    public fun distanceCm(
        a: ScreenPoint,
        b: ScreenPoint,
    ): Double {
        val dx = (a.x - b.x) * widthCm
        val dy = (a.y - b.y) * heightCm
        return kotlin.math.sqrt(dx * dx + dy * dy)
    }

    public companion object {
        public const val CM_PER_INCH: Double = 2.54
        public const val DEFAULT_CAMERA_Y_CM: Double = -0.5

        /** From Android `DisplayMetrics`; defaults assume a camera centred above the screen. */
        public fun fromDpi(
            widthPx: Int,
            heightPx: Int,
            xdpi: Double,
            ydpi: Double,
            cameraXCm: Double? = null,
            cameraYCm: Double = DEFAULT_CAMERA_Y_CM,
        ): ScreenGeometry {
            val ppcmX = xdpi / CM_PER_INCH
            val ppcmY = ydpi / CM_PER_INCH
            return ScreenGeometry(widthPx, heightPx, ppcmX, ppcmY, cameraXCm ?: (widthPx / ppcmX) / 2.0, cameraYCm)
        }
    }
}
