package app.eyegaze.domain.preprocess

/**
 * Converts a camera RGBA_8888 buffer (rows possibly padded to [rowStrideBytes]) into an upright, tightly packed
 * [RgbImage] by rotating clockwise by [rotationDegrees] (CameraX `ImageInfo.rotationDegrees`).
 *
 * The front camera's analysis buffer is the raw sensor image — it is *not* mirrored — which is exactly the
 * orientation the preprocessing spec and the training data (GazeCapture) use. Do not mirror it.
 */
public object ImageOps {
    public fun rgbaToUprightRgb(
        rgba: ByteArray,
        width: Int,
        height: Int,
        rowStrideBytes: Int,
        rotationDegrees: Int,
    ): RgbImage {
        require(width > 0 && height > 0) { "Image size must be positive" }
        require(rowStrideBytes >= width * 4) { "rowStride $rowStrideBytes < ${width * 4}" }
        require(rgba.size >= rowStrideBytes * (height - 1) + width * 4) { "Buffer too small" }
        require(rotationDegrees in setOf(0, 90, 180, 270)) { "Unsupported rotation $rotationDegrees" }
        val outW = if (rotationDegrees % 180 == 0) width else height
        val outH = if (rotationDegrees % 180 == 0) height else width
        val out = ByteArray(outW * outH * 3)
        for (y in 0 until height) {
            val rowBase = y * rowStrideBytes
            for (x in 0 until width) {
                // Destination of source pixel (x, y) after a clockwise rotation.
                val (dx, dy) =
                    when (rotationDegrees) {
                        0 -> x to y
                        90 -> (height - 1 - y) to x
                        180 -> (width - 1 - x) to (height - 1 - y)
                        else -> y to (width - 1 - x)
                    }
                val src = rowBase + x * 4
                val dst = (dy * outW + dx) * 3
                out[dst] = rgba[src]
                out[dst + 1] = rgba[src + 1]
                out[dst + 2] = rgba[src + 2]
            }
        }
        return RgbImage(outW, outH, out)
    }
}
