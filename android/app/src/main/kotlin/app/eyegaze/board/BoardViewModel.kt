package app.eyegaze.board

import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import app.eyegaze.di.DefaultDispatcher
import app.eyegaze.domain.bundle.BundleException
import app.eyegaze.domain.pipeline.CameraFrame
import app.eyegaze.domain.pipeline.GazeFrameResult
import app.eyegaze.runtime.GazeRuntime
import app.eyegaze.runtime.GazeRuntimeFactory
import app.eyegaze.runtime.RuntimeInfo
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

sealed interface BoardUiState {
    data object Loading : BoardUiState

    data class NoModel(
        val reason: String,
    ) : BoardUiState

    data class Failed(
        val reason: String,
    ) : BoardUiState

    data class Ready(
        val info: RuntimeInfo,
        val frame: GazeFrameResult? = null,
        /** Last confirmed cell and a counter so repeated selections of the same cell still animate. */
        val lastSelection: Int? = null,
        val selectionCount: Int = 0,
        val frameErrors: Int = 0,
    ) : BoardUiState
}

@HiltViewModel
class BoardViewModel
    @Inject
    constructor(
        private val factory: GazeRuntimeFactory,
        @DefaultDispatcher private val dispatcher: CoroutineDispatcher,
    ) : ViewModel() {
        private val _state = MutableStateFlow<BoardUiState>(BoardUiState.Loading)
        val state: StateFlow<BoardUiState> = _state.asStateFlow()

        @Volatile private var runtime: GazeRuntime? = null

        init {
            viewModelScope.launch(dispatcher) {
                _state.value =
                    try {
                        val created = factory.create()
                        runtime = created
                        BoardUiState.Ready(created.info)
                    } catch (e: BundleException) {
                        BoardUiState.NoModel(e.message ?: "invalid model bundle")
                    } catch (e: Exception) {
                        Log.e(TAG, "Failed to start the gaze runtime", e)
                        BoardUiState.Failed(e.message ?: e.javaClass.simpleName)
                    }
            }
        }

        /** Called on the camera analysis thread for every frame. */
        fun onFrame(frame: CameraFrame) {
            val current = runtime ?: return
            val result =
                try {
                    current.process(frame)
                } catch (e: Exception) {
                    // One bad frame must not stop tracking; count it for the diagnostics overlay.
                    Log.w(TAG, "Frame failed", e)
                    _state.update { if (it is BoardUiState.Ready) it.copy(frameErrors = it.frameErrors + 1) else it }
                    return
                }
            _state.update { state ->
                if (state !is BoardUiState.Ready) return@update state
                val selected = result.selection.selectedCell
                if (selected != null) {
                    state.copy(frame = result, lastSelection = selected, selectionCount = state.selectionCount + 1)
                } else {
                    state.copy(frame = result)
                }
            }
        }

        fun onCameraError(error: Throwable) {
            Log.e(TAG, "Camera error", error)
            _state.value = BoardUiState.Failed("Camera: ${error.message ?: error.javaClass.simpleName}")
        }

        override fun onCleared() {
            runtime?.close()
            runtime = null
        }

        private companion object {
            const val TAG = "BoardViewModel"
        }
    }
