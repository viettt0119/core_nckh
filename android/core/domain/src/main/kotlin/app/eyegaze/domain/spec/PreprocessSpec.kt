package app.eyegaze.domain.spec

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.SerializationException
import kotlinx.serialization.json.Json

/**
 * Kotlin mirror of `contracts/preprocess_spec.v1.json` (and of the `preprocess_spec.json` shipped in every
 * model bundle). Parsed at runtime so a model trained with a spec variant (e.g. 96 px eye crops) needs no
 * app code change. Validation mirrors `research/src/eyegaze/domain/spec.py`.
 */
@Serializable
public data class PreprocessSpec(
    val version: Int,
    val frame: FrameSpec,
    val sampling: SamplingSpec,
    val eye: EyeSpec,
    val face: FaceSpec,
    val geometry: GeometrySpec,
) {
    init {
        require(version == SUPPORTED_VERSION) { "Unsupported preprocess spec version $version" }
    }

    public companion object {
        public const val SUPPORTED_VERSION: Int = 1
        private val json = Json { ignoreUnknownKeys = false }

        /** Parses and validates a spec; throws [SpecException] with a readable message on any problem. */
        public fun parse(text: String): PreprocessSpec =
            try {
                json.decodeFromString(serializer(), text)
            } catch (e: SerializationException) {
                throw SpecException("Malformed preprocess spec: ${e.message}", e)
            } catch (e: IllegalArgumentException) {
                throw SpecException("Invalid preprocess spec: ${e.message}", e)
            }
    }
}

public class SpecException(
    message: String,
    cause: Throwable? = null,
) : IllegalArgumentException(message, cause)

@Serializable
public data class FrameSpec(
    @SerialName("color_order") val colorOrder: String,
    val orientation: String,
    val landmarks: String,
    @SerialName("transform_matrix") val transformMatrix: String,
) {
    init {
        require(colorOrder == "RGB") { "Only RGB frames are supported" }
    }
}

@Serializable
public data class SamplingSpec(
    val method: String,
    val supersample: Int,
    val border: String,
    @SerialName("pixel_divisor") val pixelDivisor: Double,
) {
    init {
        require(method == "bilinear") { "Unsupported sampling method $method" }
        require(border == "replicate") { "Unsupported border mode $border" }
        require(supersample in 1..4) { "supersample must be in [1, 4]" }
        require(pixelDivisor > 0) { "pixel_divisor must be positive" }
    }
}

@Serializable
public data class EyeLandmarks(
    val outer: Int,
    val inner: Int,
    @SerialName("iris_center") val irisCenter: Int,
)

@Serializable
public data class EyeSpec(
    val size: Int,
    @SerialName("width_scale") val widthScale: Double,
    @SerialName("align_roll") val alignRoll: Boolean,
    val right: EyeLandmarks,
    val left: EyeLandmarks,
    @SerialName("flip_left") val flipLeft: Boolean,
) {
    init {
        require(size >= 16) { "eye.size must be >= 16" }
        require(widthScale > 1.0) { "eye.width_scale must be > 1" }
    }
}

@Serializable
public data class FaceSpec(
    val size: Int,
    val scale: Double,
    @SerialName("align_roll") val alignRoll: Boolean,
    @SerialName("bbox_landmark_count") val bboxLandmarkCount: Int,
) {
    init {
        require(size >= 16) { "face.size must be >= 16" }
        require(scale > 0) { "face.scale must be positive" }
        require(bboxLandmarkCount in 1..LANDMARK_COUNT) { "face.bbox_landmark_count out of range" }
    }
}

@Serializable
public data class GeometrySpec(
    val dim: Int,
    val layout: List<String>,
) {
    init {
        require(layout.size == dim) { "geometry.layout has ${layout.size} entries, expected $dim" }
    }
}

/** MediaPipe Face Landmarker output size (468 mesh points + 10 iris points). */
public const val LANDMARK_COUNT: Int = 478
