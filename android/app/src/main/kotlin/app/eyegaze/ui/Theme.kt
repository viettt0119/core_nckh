package app.eyegaze.ui

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val Colors =
    lightColorScheme(
        primary = Color(0xFF2F80ED),
        secondary = Color(0xFF1E3A5F),
        tertiary = Color(0xFF27AE60),
        error = Color(0xFFD64545),
    )

@Composable
fun EyeGazeTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = Colors, content = content)
}
