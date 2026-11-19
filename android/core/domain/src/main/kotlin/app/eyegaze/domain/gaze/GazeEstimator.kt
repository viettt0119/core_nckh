package app.eyegaze.domain.gaze

import app.eyegaze.domain.calibration.RidgeModel
import app.eyegaze.domain.face.FaceKeypoints
import app.eyegaze.domain.face.FaceObservation
import app.eyegaze.domain.face.HeadPose
import app.eyegaze.domain.filter.PointFilter
import app.eyegaze.domain.geometry.GazeCm
import app.eyegaze.domain.geometry.ScreenGeometry
import app.eyegaze.domain.geometry.ScreenPoint
import app.eyegaze.domain.preprocess.ModelInputs
import app.eyegaze.domain.preprocess.Preprocessor
import app.eyegaze.domain.preprocess.RgbImage

/** Output of the network for one frame. */
public class GazePrediction(
    public val gazeCm: GazeCm,
    public val embedding: FloatArray,
)

/** Port implemented by the LiteRT adapter (`:core:ml`). */
public interface GazeModel {
    public val embeddingDim: Int

    public fun predict(inputs: ModelInputs): GazePrediction
}

/**
 * Calibration features, identical to `research/.../evaluate.py::calibration_features`:
 * `[embedding…, gaze_cm.x, gaze_cm.y, pitch, yaw, roll]`.
 */
public fun calibrationFeatures(
    prediction: GazePrediction,
    pose: HeadPose,
): DoubleArray {
    val e = prediction.embedding
    val out = DoubleArray(e.size + 5)
    for (i in e.indices) out[i] = e[i].toDouble()
    out[e.size] = prediction.gazeCm.x
    out[e.size + 1] = prediction.gazeCm.y
    out[e.size + 2] = pose.pitch
    out[e.size + 3] = pose.yaw
    out[e.size + 4] = pose.roll
    return out
}

/** Maps a network prediction to normalized screen coordinates. */
public fun interface ScreenMapper {
    public fun map(
        prediction: GazePrediction,
        pose: HeadPose,
    ): ScreenPoint
}

/** Before calibration: physical geometry only (camera assumed centred above the screen). */
public class UncalibratedMapper(
    private val screen: ScreenGeometry,
) : ScreenMapper {
    override fun map(
        prediction: GazePrediction,
        pose: HeadPose,
    ): ScreenPoint = screen.cameraCmToScreen(prediction.gazeCm)
}

/** After calibration: the per-user ridge model maps features straight to the screen. */
public class CalibratedMapper(
    private val model: RidgeModel,
) : ScreenMapper {
    init {
        require(model.targetDim == 2) { "Calibration model must predict 2 screen coordinates" }
    }

    override fun map(
        prediction: GazePrediction,
        pose: HeadPose,
    ): ScreenPoint {
        val p = model.predict(calibrationFeatures(prediction, pose))
        return ScreenPoint(p[0], p[1])
    }
}

public class GazeEstimate(
    public val keypoints: FaceKeypoints,
    public val prediction: GazePrediction,
    public val rawScreen: ScreenPoint,
    public val smoothedScreen: ScreenPoint,
)

/** Frame → preprocessing → network → screen mapping → smoothing. */
public class GazeEstimator(
    private val preprocessor: Preprocessor,
    private val model: GazeModel,
    @Volatile public var mapper: ScreenMapper,
    private val filter: PointFilter = PointFilter(),
) {
    public fun estimate(
        image: RgbImage,
        observation: FaceObservation,
        timeNanos: Long,
    ): GazeEstimate {
        val keypoints = preprocessor.keypoints(observation)
        val prediction = model.predict(preprocessor.process(image, keypoints))
        val raw = mapper.map(prediction, keypoints.pose)
        return GazeEstimate(keypoints, prediction, raw, filter.filter(raw, timeNanos))
    }

    /** Call when tracking is interrupted (face lost) so smoothing does not bridge the gap. */
    public fun reset() {
        filter.reset()
    }
}
