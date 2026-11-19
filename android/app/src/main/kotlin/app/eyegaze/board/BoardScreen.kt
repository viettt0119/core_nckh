package app.eyegaze.board

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import app.eyegaze.R
import app.eyegaze.camera.FrontCameraSource
import app.eyegaze.domain.interaction.BlinkPhase
import kotlinx.coroutines.awaitCancellation
import java.util.Locale

@Composable
fun AppRoot() {
    val context = LocalContext.current
    var granted by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED,
        )
    }
    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted = it }
    if (granted) {
        BoardRoute()
    } else {
        Message(
            title = stringResource(R.string.camera_permission_title),
            body = stringResource(R.string.camera_permission_body),
            action =
                stringResource(
                    R.string.camera_permission_grant,
                ) to { launcher.launch(Manifest.permission.CAMERA) },
        )
    }
}

@Composable
fun BoardRoute(viewModel: BoardViewModel = hiltViewModel()) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val camera = remember { FrontCameraSource(context.applicationContext) }
    val ready = state is BoardUiState.Ready
    LaunchedEffect(ready, lifecycleOwner) {
        if (!ready) return@LaunchedEffect
        val handle =
            try {
                camera.bind(lifecycleOwner, viewModel::onFrame, viewModel::onCameraError)
            } catch (e: Exception) {
                viewModel.onCameraError(e)
                return@LaunchedEffect
            }
        try {
            awaitCancellation()
        } finally {
            handle.close()
        }
    }
    when (val s = state) {
        BoardUiState.Loading -> Loading()
        is BoardUiState.NoModel ->
            Message(
                stringResource(R.string.no_model_title),
                stringResource(R.string.no_model_body, s.reason),
            )
        is BoardUiState.Failed -> Message(stringResource(R.string.failed_title), s.reason)
        is BoardUiState.Ready -> Board(s)
    }
}

@Composable
private fun Board(state: BoardUiState.Ready) {
    var showDiagnostics by rememberSaveable { mutableStateOf(false) }
    val grid = state.info.grid
    val selection = state.frame?.selection
    val flash = remember { Animatable(0f) }
    LaunchedEffect(state.selectionCount) {
        if (state.selectionCount > 0) {
            flash.snapTo(1f)
            flash.animateTo(0f, tween(durationMillis = 700))
        }
    }
    val primary = MaterialTheme.colorScheme.primary
    val confirm = MaterialTheme.colorScheme.tertiary
    Box(
        Modifier
            .fillMaxSize()
            .background(Color.White)
            .pointerInput(Unit) { detectTapGestures(onLongPress = { showDiagnostics = !showDiagnostics }) },
    ) {
        Column(Modifier.fillMaxSize()) {
            for (row in 0 until grid.rows) {
                Row(Modifier.fillMaxWidth().weight(1f)) {
                    for (col in 0 until grid.cols) {
                        val cell = row * grid.cols + col
                        val hovered = selection?.hoverCell == cell
                        val locked = selection?.lockedCell == cell
                        val flashing = state.lastSelection == cell
                        val background =
                            when {
                                flashing && flash.value > 0f -> confirm.copy(alpha = 0.25f + 0.6f * flash.value)
                                hovered -> primary.copy(alpha = 0.22f)
                                else -> Color(0xFFF4F6F9)
                            }
                        Box(
                            Modifier
                                .weight(1f)
                                .fillMaxSize()
                                .padding(6.dp)
                                .background(background, MaterialTheme.shapes.large)
                                .border(
                                    if (locked) 5.dp else 1.dp,
                                    if (locked) primary else Color(0xFFD5DBE3),
                                    MaterialTheme.shapes.large,
                                ),
                            contentAlignment = Alignment.Center,
                        ) {
                            Text("${cell + 1}", fontSize = 48.sp, color = MaterialTheme.colorScheme.secondary)
                            val closing = selection?.takeIf { locked && it.phase != BlinkPhase.TRACKING }
                            if (closing != null) {
                                CircularProgressIndicator(
                                    progress = { closing.blinkProgress.toFloat() },
                                    modifier = Modifier.size(120.dp),
                                    strokeWidth = 8.dp,
                                )
                            }
                        }
                    }
                }
            }
        }
        state.frame?.gaze?.let { gaze ->
            Canvas(Modifier.fillMaxSize()) {
                val center =
                    androidx.compose.ui.geometry.Offset(
                        (gaze.x * size.width).toFloat(),
                        (gaze.y * size.height).toFloat(),
                    )
                drawCircle(primary.copy(alpha = 0.35f), radius = 28f, center = center)
                drawCircle(primary, radius = 28f, center = center, style = Stroke(width = 4f))
            }
        }
        Column(
            Modifier.align(Alignment.BottomCenter).padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            if (selection?.faceVisible == false) {
                Banner(stringResource(R.string.no_face), MaterialTheme.colorScheme.error)
            } else if (state.lastSelection != null) {
                Banner(stringResource(R.string.selected_cell, state.lastSelection + 1), confirm)
            } else {
                Banner(stringResource(R.string.hint), MaterialTheme.colorScheme.secondary)
            }
        }
        if (showDiagnostics) Diagnostics(state, Modifier.align(Alignment.TopStart))
    }
}

@Composable
private fun Diagnostics(
    state: BoardUiState.Ready,
    modifier: Modifier,
) {
    val m = state.frame?.metrics
    val pose = state.frame?.pose
    val lines =
        listOfNotNull(
            with(state.info) { "model $modelRun @$modelEpoch · $accelerator · spec ${specSha256.take(8)}" },
            m?.let {
                String.format(
                    Locale.US,
                    "%.1f fps · landmarks %.1f ms · gaze %.1f ms · total %.1f ms",
                    it.fps,
                    it.landmarkMs,
                    it.gazeMs,
                    it.totalMs,
                )
            },
            pose?.let { String.format(Locale.US, "pose pitch %.2f yaw %.2f roll %.2f rad", it.pitch, it.yaw, it.roll) },
            state.frame?.rawGaze?.let { String.format(Locale.US, "raw gaze %.3f, %.3f", it.x, it.y) },
            state.frame?.selection?.let { "phase ${it.phase} · hover ${it.hoverCell} · locked ${it.lockedCell}" },
            "frame errors ${state.frameErrors}",
        )
    Column(modifier.padding(12.dp).background(Color(0xCC0B1F33), MaterialTheme.shapes.small).padding(10.dp)) {
        lines.forEach { Text(it, color = Color.White, fontFamily = FontFamily.Monospace, fontSize = 12.sp) }
    }
}

@Composable
private fun Banner(
    text: String,
    color: Color,
) {
    Text(
        text,
        color = Color.White,
        textAlign = TextAlign.Center,
        modifier =
            Modifier
                .background(
                    color.copy(alpha = 0.9f),
                    MaterialTheme.shapes.medium,
                ).padding(horizontal = 16.dp, vertical = 10.dp),
    )
}

@Composable
private fun Loading() {
    Column(
        Modifier.fillMaxSize(),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        CircularProgressIndicator()
        Spacer(Modifier.height(16.dp))
        Text(stringResource(R.string.loading))
    }
}

@Composable
private fun Message(
    title: String,
    body: String,
    action: Pair<String, () -> Unit>? = null,
) {
    Column(
        Modifier.fillMaxSize().padding(32.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(title, style = MaterialTheme.typography.headlineSmall, textAlign = TextAlign.Center)
        Spacer(Modifier.height(12.dp))
        Text(body, style = MaterialTheme.typography.bodyLarge, textAlign = TextAlign.Center)
        action?.let { (label, onClick) ->
            Spacer(Modifier.height(24.dp))
            Button(onClick = onClick) { Text(label) }
        }
    }
}
