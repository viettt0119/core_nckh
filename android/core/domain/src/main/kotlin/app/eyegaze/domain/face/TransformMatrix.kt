package app.eyegaze.domain.face

/**
 * MediaPipe's Android API returns the facial transformation matrix as 16 floats in **column-major** order,
 * while the Python API (used to build the training set) converts the same data to a row-major matrix.
 * [FaceObservation] uses row-major, so adapters must convert with this function.
 */
public fun columnMajorToRowMajor(columnMajor: FloatArray): DoubleArray {
    require(columnMajor.size == 16) { "Expected 16 values, got ${columnMajor.size}" }
    return DoubleArray(16) { i ->
        val row = i / 4
        val col = i % 4
        columnMajor[col * 4 + row].toDouble()
    }
}
