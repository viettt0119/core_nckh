package app.eyegaze.domain.bundle

import app.eyegaze.domain.spec.PreprocessSpec
import app.eyegaze.domain.spec.SpecException
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.SerializationException
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import java.security.MessageDigest

/** The fields of `model_card.json` the app relies on (unknown fields are ignored for forward compatibility). */
@Serializable
public data class ModelCard(
    val format: String,
    @SerialName("model_file") val modelFile: String,
    @SerialName("model_sha256") val modelSha256: String,
    @SerialName("spec_file") val specFile: String,
    @SerialName("spec_sha256") val specSha256: String,
    @SerialName("model_io_version") val modelIoVersion: Int,
    @SerialName("embedding_dim") val embeddingDim: Int,
    val precision: String,
    @SerialName("source_run") val sourceRun: String,
    @SerialName("source_epoch") val sourceEpoch: Int,
    @SerialName("test_metrics") val testMetrics: JsonElement? = null,
)

public class BundleException(
    message: String,
    cause: Throwable? = null,
) : IllegalStateException(message, cause)

/** A bundle whose hashes matched its model card: safe to run. */
public class VerifiedBundle(
    public val card: ModelCard,
    public val spec: PreprocessSpec,
    public val modelBytes: ByteArray,
)

/**
 * Start-up verification of the model bundle produced by `eyegaze export` (ADR-0006): the model and the
 * preprocessing spec must be byte-identical to what the model card describes, otherwise the app refuses to run.
 */
public object BundleVerifier {
    public const val CARD_FILE: String = "model_card.json"
    public const val CARD_FORMAT: String = "eyegaze-model-card-v1"
    public const val SUPPORTED_MODEL_IO_VERSION: Int = 1
    private val json = Json { ignoreUnknownKeys = true }

    /** [read] returns the bytes of a bundle file by name, or null if it does not exist. */
    public fun verify(read: (String) -> ByteArray?): VerifiedBundle {
        val cardBytes = read(CARD_FILE) ?: throw BundleException("No model bundle installed ($CARD_FILE missing)")
        val card =
            try {
                json.decodeFromString(ModelCard.serializer(), cardBytes.decodeToString())
            } catch (e: SerializationException) {
                throw BundleException("Malformed $CARD_FILE: ${e.message}", e)
            }
        if (card.format != CARD_FORMAT) throw BundleException("Unsupported model card format ${card.format}")
        if (card.modelIoVersion != SUPPORTED_MODEL_IO_VERSION) {
            throw BundleException("Model I/O version ${card.modelIoVersion} is not supported by this app")
        }
        val model = read(card.modelFile) ?: throw BundleException("${card.modelFile} missing from bundle")
        val specBytes = read(card.specFile) ?: throw BundleException("${card.specFile} missing from bundle")
        checkHash(card.modelFile, model, card.modelSha256)
        checkHash(card.specFile, specBytes, card.specSha256)
        val spec =
            try {
                PreprocessSpec.parse(specBytes.decodeToString())
            } catch (e: SpecException) {
                throw BundleException("Bundled spec rejected: ${e.message}", e)
            }
        return VerifiedBundle(card, spec, model)
    }

    public fun sha256(bytes: ByteArray): String =
        MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }

    private fun checkHash(
        name: String,
        bytes: ByteArray,
        expected: String,
    ) {
        val actual = sha256(bytes)
        if (!actual.equals(expected, ignoreCase = true)) {
            throw BundleException("$name SHA-256 $actual does not match model card $expected")
        }
    }
}
