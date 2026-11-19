package app.eyegaze.ml

import android.content.res.AssetManager
import app.eyegaze.domain.bundle.BundleVerifier
import app.eyegaze.domain.bundle.VerifiedBundle
import java.io.FileNotFoundException

/** Loads and verifies the model bundle packaged under `assets/<dir>/` (see `eyegaze export`). */
public class AssetBundleSource(
    private val assets: AssetManager,
    private val dir: String = DEFAULT_DIR,
) {
    public fun load(): VerifiedBundle =
        BundleVerifier.verify { name ->
            try {
                assets.open("$dir/$name").use { it.readBytes() }
            } catch (e: FileNotFoundException) {
                null
            }
        }

    public companion object {
        public const val DEFAULT_DIR: String = "model"
    }
}
