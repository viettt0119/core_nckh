package app.eyegaze.ml

import android.util.Log
import app.eyegaze.domain.gaze.GazeModel
import app.eyegaze.domain.gaze.GazePrediction
import app.eyegaze.domain.geometry.GazeCm
import app.eyegaze.domain.preprocess.ModelInputs
import app.eyegaze.domain.spec.PreprocessSpec
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Which LiteRT backend ended up running the model. */
public enum class Accelerator { GPU, CPU }

/**
 * Signature contract of `contracts/model_io.v1.json`. Tensors are bound by name, never by index:
 * converters are free to reorder them.
 */
public object GazeSignature {
    public const val KEY: String = "serving_default"
    public val INPUTS: Set<String> = setOf("right_eye", "left_eye", "face", "geometry")
    public val OUTPUTS: Set<String> = setOf("gaze_cm", "embedding")

    /** Returns a description of every violation; empty means the model honours the contract. */
    public fun violations(
        signatureKeys: Collection<String>,
        inputs: Collection<String>,
        outputs: Collection<String>,
    ): List<String> {
        if (KEY !in signatureKeys) return listOf("missing signature '$KEY' (found $signatureKeys)")
        return (INPUTS - inputs.toSet()).map { "missing input '$it'" } +
            (inputs.toSet() - INPUTS).map { "unexpected input '$it'" } +
            (OUTPUTS - outputs.toSet()).map { "missing output '$it'" } +
            (outputs.toSet() - OUTPUTS).map { "unexpected output '$it'" }
    }
}

public class ModelLoadException(
    message: String,
    cause: Throwable? = null,
) : IllegalStateException(message, cause)

/**
 * [GazeModel] backed by a LiteRT interpreter. Tries the GPU delegate (FP16 compute allowed) and falls back to
 * the CPU (XNNPACK) if the device or the graph does not support it. Not thread-safe: call from one thread
 * (the camera analysis thread).
 */
public class LiteRtGazeModel private constructor(
    private val interpreter: Interpreter,
    private val gpu: GpuDelegate?,
    private val spec: PreprocessSpec,
    override val embeddingDim: Int,
    public val accelerator: Accelerator,
) : GazeModel,
    Closeable {
    private val eyeBuffer = { floatBuffer(spec.eye.size * spec.eye.size * 3) }
    private val rightEye = eyeBuffer()
    private val leftEye = eyeBuffer()
    private val face = floatBuffer(spec.face.size * spec.face.size * 3)
    private val geometry = floatBuffer(spec.geometry.dim)
    private val gazeOut = floatBuffer(2)
    private val embeddingOut = floatBuffer(embeddingDim)

    override fun predict(inputs: ModelInputs): GazePrediction {
        fill(rightEye, inputs.rightEye)
        fill(leftEye, inputs.leftEye)
        fill(face, inputs.face)
        fill(geometry, inputs.geometry)
        gazeOut.rewind()
        embeddingOut.rewind()
        interpreter.runSignature(
            mapOf("right_eye" to rightEye, "left_eye" to leftEye, "face" to face, "geometry" to geometry),
            mapOf("gaze_cm" to gazeOut, "embedding" to embeddingOut),
            GazeSignature.KEY,
        )
        gazeOut.rewind()
        embeddingOut.rewind()
        val g = gazeOut.asFloatBuffer()
        val embedding = FloatArray(embeddingDim).also { embeddingOut.asFloatBuffer().get(it) }
        return GazePrediction(GazeCm(g.get(0).toDouble(), g.get(1).toDouble()), embedding)
    }

    override fun close() {
        interpreter.close()
        gpu?.close()
    }

    public companion object {
        private const val TAG = "LiteRtGazeModel"

        public fun create(
            modelBytes: ByteArray,
            spec: PreprocessSpec,
            embeddingDim: Int,
            preferGpu: Boolean = true,
            cpuThreads: Int = 4,
        ): LiteRtGazeModel {
            val buffer = ByteBuffer.allocateDirect(modelBytes.size).order(ByteOrder.nativeOrder())
            buffer.put(modelBytes).rewind()
            if (preferGpu) {
                tryGpu(buffer, spec, embeddingDim)?.let { return it }
            }
            val interpreter = Interpreter(buffer, Interpreter.Options().setNumThreads(cpuThreads))
            return validated(interpreter, null, spec, embeddingDim, Accelerator.CPU)
        }

        private fun tryGpu(
            buffer: ByteBuffer,
            spec: PreprocessSpec,
            embeddingDim: Int,
        ): LiteRtGazeModel? {
            var delegate: GpuDelegate? = null
            return try {
                val compatibility = CompatibilityList()
                if (!compatibility.isDelegateSupportedOnThisDevice) return null
                delegate = GpuDelegate(compatibility.bestOptionsForThisDevice.setPrecisionLossAllowed(true))
                val interpreter = Interpreter(buffer, Interpreter.Options().addDelegate(delegate))
                validated(interpreter, delegate, spec, embeddingDim, Accelerator.GPU).also { it.warmUp() }
            } catch (e: Exception) {
                Log.w(TAG, "GPU delegate unavailable, falling back to CPU", e)
                delegate?.close()
                buffer.rewind()
                null
            }
        }

        private fun validated(
            interpreter: Interpreter,
            gpu: GpuDelegate?,
            spec: PreprocessSpec,
            embeddingDim: Int,
            accelerator: Accelerator,
        ): LiteRtGazeModel {
            val problems =
                GazeSignature.violations(
                    interpreter.signatureKeys.toList(),
                    interpreter.getSignatureInputs(GazeSignature.KEY).toList(),
                    interpreter.getSignatureOutputs(GazeSignature.KEY).toList(),
                )
            if (problems.isNotEmpty()) {
                interpreter.close()
                gpu?.close()
                throw ModelLoadException("Model violates contracts/model_io.v1.json: ${problems.joinToString("; ")}")
            }
            return LiteRtGazeModel(interpreter, gpu, spec, embeddingDim, accelerator)
        }

        private fun floatBuffer(count: Int): ByteBuffer =
            ByteBuffer.allocateDirect(count * 4).order(ByteOrder.nativeOrder())

        private fun fill(
            buffer: ByteBuffer,
            values: FloatArray,
        ) {
            require(values.size * 4 == buffer.capacity()) {
                "Input has ${values.size} values, tensor expects ${buffer.capacity() / 4}"
            }
            buffer.rewind()
            buffer.asFloatBuffer().put(values)
            buffer.rewind()
        }
    }

    private fun warmUp() {
        predict(
            ModelInputs(
                FloatArray(rightEye.capacity() / 4),
                FloatArray(leftEye.capacity() / 4),
                FloatArray(face.capacity() / 4),
                FloatArray(geometry.capacity() / 4),
            ),
        )
    }
}
