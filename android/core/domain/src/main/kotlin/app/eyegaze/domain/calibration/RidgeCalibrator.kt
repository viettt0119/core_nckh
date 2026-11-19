package app.eyegaze.domain.calibration

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Per-user calibration: standardize → linear map, fitted by closed-form ridge regression with an
 * unpenalized bias, λ chosen by leave-one-*target*-out cross-validation (frames on the same calibration
 * dot are correlated, so leaving out single frames would under-estimate error).
 *
 * Port of `research/src/eyegaze/domain/calibration.py`; `contracts/golden/calibration.json` enforces agreement.
 */
public class RidgeModel(
    public val featureMean: DoubleArray,
    public val featureStd: DoubleArray,
    /** (D + 1) × T row-major; the last row is the bias. */
    public val weights: Array<DoubleArray>,
    public val lambda: Double,
    public val cvError: Double,
) {
    public val featureDim: Int get() = featureMean.size
    public val targetDim: Int get() = weights[0].size

    init {
        require(featureStd.size == featureMean.size) { "mean/std size mismatch" }
        require(weights.size == featureMean.size + 1 && weights.all { it.size == weights[0].size }) {
            "weights must be (D + 1) x T"
        }
    }

    public fun predict(features: DoubleArray): DoubleArray {
        require(features.size == featureDim) { "Expected $featureDim features, got ${features.size}" }
        val out = DoubleArray(targetDim)
        for (t in 0 until targetDim) {
            var acc = weights[featureDim][t]
            for (d in 0 until featureDim) {
                acc += (features[d] - featureMean[d]) / featureStd[d] * weights[d][t]
            }
            out[t] = acc
        }
        return out
    }
}

public class CalibrationException(
    message: String,
) : IllegalArgumentException(message)

public object RidgeCalibrator {
    public val DEFAULT_LAMBDAS: List<Double> = listOf(1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
    public const val MIN_TARGET_GROUPS: Int = 3
    private const val STD_FLOOR = 1e-6

    public fun fit(
        features: List<DoubleArray>,
        targets: List<DoubleArray>,
        groups: IntArray,
        lambdas: List<Double> = DEFAULT_LAMBDAS,
    ): RidgeModel {
        if (features.size != targets.size || features.size != groups.size) {
            throw CalibrationException(
                "Length mismatch: features=${features.size}, targets=${targets.size}, groups=${groups.size}",
            )
        }
        if (features.isEmpty()) throw CalibrationException("No calibration samples")
        val d = features[0].size
        val t = targets[0].size
        if (features.any { it.size != d } ||
            targets.any { it.size != t }
        ) {
            throw CalibrationException("Ragged input arrays")
        }
        if (features.any { row -> row.any { !it.isFinite() } } || targets.any { row -> row.any { !it.isFinite() } }) {
            throw CalibrationException("features and targets must be finite")
        }
        if (lambdas.isEmpty() || lambdas.any { it <= 0 }) throw CalibrationException("lambdas must be positive")
        val uniqueGroups = groups.distinct().sorted()
        if (uniqueGroups.size < MIN_TARGET_GROUPS) {
            throw CalibrationException(
                "Need samples from at least $MIN_TARGET_GROUPS calibration targets, got ${uniqueGroups.size}",
            )
        }

        var bestLambda = lambdas[0]
        var bestError = Double.POSITIVE_INFINITY
        for (lam in lambdas) {
            var foldSum = 0.0
            for (heldOut in uniqueGroups) {
                val train = groups.indices.filter { groups[it] != heldOut }
                val test = groups.indices.filter { groups[it] == heldOut }
                val (mean, std) = standardize(features, train)
                val w = solve(features, targets, train, mean, std, lam)
                var errSum = 0.0
                for (i in test) {
                    val p = predictRaw(features[i], mean, std, w)
                    var sq = 0.0
                    for (k in 0 until t) sq += (p[k] - targets[i][k]) * (p[k] - targets[i][k])
                    errSum += sqrt(sq)
                }
                foldSum += errSum / test.size
            }
            val err = foldSum / uniqueGroups.size
            if (err < bestError) {
                bestError = err
                bestLambda = lam
            }
        }
        val all = groups.indices.toList()
        val (mean, std) = standardize(features, all)
        return RidgeModel(mean, std, solve(features, targets, all, mean, std, bestLambda), bestLambda, bestError)
    }

    private fun standardize(
        x: List<DoubleArray>,
        rows: List<Int>,
    ): Pair<DoubleArray, DoubleArray> {
        val d = x[0].size
        val mean = DoubleArray(d)
        for (i in rows) for (j in 0 until d) mean[j] += x[i][j]
        for (j in 0 until d) mean[j] /= rows.size
        val std = DoubleArray(d)
        for (i in rows) for (j in 0 until d) std[j] += (x[i][j] - mean[j]) * (x[i][j] - mean[j])
        for (j in 0 until d) std[j] = max(sqrt(std[j] / rows.size), STD_FLOOR) // population std, as numpy
        return mean to std
    }

    private fun predictRaw(
        x: DoubleArray,
        mean: DoubleArray,
        std: DoubleArray,
        w: Array<DoubleArray>,
    ): DoubleArray {
        val d = x.size
        val out = DoubleArray(w[0].size)
        for (t in out.indices) {
            var acc = w[d][t]
            for (j in 0 until d) acc += (x[j] - mean[j]) / std[j] * w[j][t]
            out[t] = acc
        }
        return out
    }

    /** Solves (XᵀX + λI′) W = XᵀY on standardized features with a bias column; I′ leaves the bias unpenalized. */
    private fun solve(
        x: List<DoubleArray>,
        y: List<DoubleArray>,
        rows: List<Int>,
        mean: DoubleArray,
        std: DoubleArray,
        lambda: Double,
    ): Array<DoubleArray> {
        val d = x[0].size
        val n = d + 1
        val t = y[0].size
        val gram = Array(n) { DoubleArray(n) }
        val rhs = Array(n) { DoubleArray(t) }
        val z = DoubleArray(n)
        for (i in rows) {
            for (j in 0 until d) z[j] = (x[i][j] - mean[j]) / std[j]
            z[d] = 1.0
            for (a in 0 until n) {
                for (b in 0 until n) gram[a][b] += z[a] * z[b]
                for (k in 0 until t) rhs[a][k] += z[a] * y[i][k]
            }
        }
        for (j in 0 until d) gram[j][j] += lambda
        return gaussianSolve(gram, rhs)
    }

    /** Gaussian elimination with partial pivoting; solves A X = B in place. */
    internal fun gaussianSolve(
        a: Array<DoubleArray>,
        b: Array<DoubleArray>,
    ): Array<DoubleArray> {
        val n = a.size
        val t = b[0].size
        for (col in 0 until n) {
            var pivot = col
            for (r in col + 1 until n) if (abs(a[r][col]) > abs(a[pivot][col])) pivot = r
            if (abs(a[pivot][col]) < 1e-12) throw CalibrationException("Singular calibration system")
            if (pivot != col) {
                a[col] = a[pivot].also { a[pivot] = a[col] }
                b[col] = b[pivot].also { b[pivot] = b[col] }
            }
            for (r in col + 1 until n) {
                val f = a[r][col] / a[col][col]
                if (f == 0.0) continue
                for (c in col until n) a[r][c] -= f * a[col][c]
                for (k in 0 until t) b[r][k] -= f * b[col][k]
            }
        }
        val xOut = Array(n) { DoubleArray(t) }
        for (r in n - 1 downTo 0) {
            for (k in 0 until t) {
                var acc = b[r][k]
                for (c in r + 1 until n) acc -= a[r][c] * xOut[c][k]
                xOut[r][k] = acc / a[r][r]
            }
        }
        return xOut
    }
}
