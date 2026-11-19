package app.eyegaze.domain.golden

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.double
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.zip.GZIPInputStream

/** Access to `contracts/` (path injected by Gradle) and helpers to decode the golden fixture formats. */
object Golden {
    val contracts: File =
        File(
            requireNotNull(System.getProperty("eyegaze.contracts")) { "eyegaze.contracts system property not set" },
        )
    val dir: File = File(contracts, "golden")

    fun json(name: String): JsonObject = Json.parseToJsonElement(File(dir, name).readText()).jsonObject

    fun gunzip(name: String): ByteArray = GZIPInputStream(File(dir, name).inputStream()).use { it.readBytes() }

    /** Inverse of the delta-x encoding: running sum mod 256 along x, per row and channel. */
    fun decodeDeltaX(
        encoded: ByteArray,
        width: Int,
        height: Int,
    ): ByteArray {
        val out = ByteArray(encoded.size)
        for (y in 0 until height) {
            for (c in 0 until 3) {
                var acc = 0
                for (x in 0 until width) {
                    val i = (y * width + x) * 3 + c
                    acc = (acc + (encoded[i].toInt() and 0xFF)) and 0xFF
                    out[i] = acc.toByte()
                }
            }
        }
        return out
    }

    fun floatsLE(bytes: ByteArray): FloatArray {
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(buf.remaining()).also { buf.get(it) }
    }

    fun cases(): List<String> = json("index.json")["cases"]!!.jsonArray.map { it.jsonPrimitive.content }

    fun tolerance(key: String): Double = json("index.json")["tolerances"]!!.jsonObject[key]!!.jsonPrimitive.double
}

val JsonElement.d: Double get() = jsonPrimitive.double
val JsonElement.i: Int get() = jsonPrimitive.int

fun JsonElement.doubles(): DoubleArray = jsonArray.map { it.d }.toDoubleArray()

fun JsonElement.matrix(): List<DoubleArray> = jsonArray.map { it.doubles() }

fun JsonElement.flatDoubles(): DoubleArray = (this as JsonArray).flatMap { row -> row.jsonArray.map { it.d } }.toDoubleArray()
