package app.eyegaze.ml

import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class GazeSignatureTest {
    private val inputs = listOf("face", "geometry", "left_eye", "right_eye")
    private val outputs = listOf("embedding", "gaze_cm")

    @Test
    fun conformingSignatureHasNoViolations() {
        assertTrue(GazeSignature.violations(listOf("serving_default"), inputs, outputs).isEmpty())
    }

    @Test
    fun violationsAreListed() {
        assertEquals(1, GazeSignature.violations(listOf("other"), inputs, outputs).size)
        val problems = GazeSignature.violations(listOf("serving_default"), inputs - "face" + "extra", outputs - "embedding")
        assertTrue("missing input 'face'" in problems)
        assertTrue("unexpected input 'extra'" in problems)
        assertTrue("missing output 'embedding'" in problems)
    }
}
