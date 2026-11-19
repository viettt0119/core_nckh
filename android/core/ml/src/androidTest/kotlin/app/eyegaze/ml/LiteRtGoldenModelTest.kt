package app.eyegaze.ml

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import app.eyegaze.domain.preprocess.ModelInputs
import app.eyegaze.domain.spec.PreprocessSpec
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.double
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Test
import org.junit.runner.RunWith
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.zip.GZIPInputStream
import kotlin.math.abs
import kotlin.test.assertTrue

/** On-device check that LiteRT (CPU and, where available, GPU) reproduces the golden model outputs. */
@RunWith(AndroidJUnit4::class)
class LiteRtGoldenModelTest {
    private val assets = InstrumentationRegistry.getInstrumentation().context.assets

    private fun text(path: String) = assets.open(path).use { it.readBytes().decodeToString() }

    private fun floats(path: String): FloatArray {
        val bytes = GZIPInputStream(assets.open(path)).use { it.readBytes() }
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(buffer.remaining()).also { buffer.get(it) }
    }

    private fun check(
        preferGpu: Boolean,
        tolerance: Double,
    ) {
        val spec = PreprocessSpec.parse(text("preprocess_spec.v1.json"))
        val expected = Json.parseToJsonElement(text("golden/golden_model_expected.json")).jsonObject
        val modelBytes = assets.open("golden/golden_model.tflite").use { it.readBytes() }
        val embeddingDim = expected["embedding_dim"]!!.jsonPrimitive.int
        LiteRtGazeModel.create(modelBytes, spec, embeddingDim, preferGpu).use { model ->
            for ((name, outputs) in expected["cases"]!!.jsonObject) {
                val meta = Json.parseToJsonElement(text("golden/$name.json")).jsonObject["expected"]!!.jsonObject
                val flat = floats("golden/" + meta["crops"]!!.jsonObject["file"]!!.jsonPrimitive.content)
                val eye = spec.eye.size * spec.eye.size * 3
                val inputs =
                    ModelInputs(
                        flat.copyOfRange(0, eye),
                        flat.copyOfRange(eye, 2 * eye),
                        flat.copyOfRange(2 * eye, flat.size),
                        meta["geometry"]!!.jsonArray.map { it.jsonPrimitive.double.toFloat() }.toFloatArray(),
                    )
                val prediction = model.predict(inputs)
                val gaze = outputs.jsonObject["gaze_cm"]!!.jsonArray.map { it.jsonPrimitive.double }
                assertTrue(abs(gaze[0] - prediction.gazeCm.x) <= tolerance, "$name gaze x (${model.accelerator})")
                assertTrue(abs(gaze[1] - prediction.gazeCm.y) <= tolerance, "$name gaze y (${model.accelerator})")
                val embedding = outputs.jsonObject["embedding"]!!.jsonArray.map { it.jsonPrimitive.double }
                embedding.forEachIndexed {
                    i,
                    v,
                    ->
                    assertTrue(abs(v - prediction.embedding[i]) <= tolerance, "$name emb[$i]")
                }
            }
        }
    }

    @Test
    fun cpuMatchesGolden() = check(preferGpu = false, tolerance = 1e-4)

    @Test
    fun gpuOrFallbackMatchesGoldenWithinFp16Tolerance() = check(preferGpu = true, tolerance = 5e-2)
}
