import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import java.net.URI
import java.security.MessageDigest

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.ksp)
    alias(libs.plugins.hilt)
}

android {
    namespace = "app.eyegaze"
    compileSdk = 35

    defaultConfig {
        applicationId = "app.eyegaze"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "2.0.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }
    buildTypes {
        release {
            // R8 is left off until keep rules for MediaPipe/LiteRT are validated on a device.
            isMinifyEnabled = false
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    buildFeatures { compose = true }
    androidResources {
        noCompress += listOf("tflite", "task")
    }
    lint {
        abortOnError = true
        warningsAsErrors = false
        checkDependencies = true
    }
    testOptions {
        unitTests.isReturnDefaultValues = true
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_17)
        allWarningsAsErrors.set(true)
    }
}

// --------------------------------------------------------------------------- packaged assets
//
// face_landmarker.task is downloaded once and pinned by SHA-256: it must be the exact model used to landmark
// the training set (ADR-0002). The gaze model bundle comes from `eyegaze export`; pass its directory with
// -Peyegaze.bundle=/path/to/bundle (or EYEGAZE_BUNDLE). Without it the app builds and explains how to add one.

abstract class DownloadVerifiedFile : DefaultTask() {
    @get:Input abstract val url: Property<String>

    @get:Input abstract val sha256: Property<String>

    @get:OutputFile abstract val output: RegularFileProperty

    @TaskAction
    fun download() {
        val target = output.get().asFile
        val bytes = URI(url.get()).toURL().openStream().use { it.readBytes() }
        val actual = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
        check(actual == sha256.get()) { "SHA-256 mismatch for ${url.get()}: $actual != ${sha256.get()}" }
        target.parentFile.mkdirs()
        target.writeBytes(bytes)
    }
}

abstract class AssembleGazeAssets : DefaultTask() {
    @get:InputFile abstract val landmarker: RegularFileProperty

    @get:Optional @get:InputDirectory
    abstract val bundle: DirectoryProperty

    @get:OutputDirectory abstract val outputDir: DirectoryProperty

    @TaskAction
    fun assemble() {
        val out = outputDir.get().asFile
        out.deleteRecursively()
        out.mkdirs()
        landmarker.get().asFile.copyTo(out.resolve("face_landmarker.task"))
        if (bundle.isPresent) {
            val source = bundle.get().asFile
            val target = out.resolve("model").apply { mkdirs() }
            for (name in listOf("model_card.json", "gaze_model.tflite", "preprocess_spec.json")) {
                val file = source.resolve(name)
                check(file.isFile) { "Model bundle $source is missing $name (export it with `eyegaze export`)" }
                file.copyTo(target.resolve(name))
            }
        }
    }
}

val downloadFaceLandmarker =
    tasks.register<DownloadVerifiedFile>("downloadFaceLandmarker") {
        url.set(
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        )
        sha256.set("64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff")
        output.set(layout.buildDirectory.file("downloads/face_landmarker.task"))
    }

val bundlePath: String? =
    providers.gradleProperty("eyegaze.bundle").orElse(providers.environmentVariable("EYEGAZE_BUNDLE")).orNull

val assembleGazeAssets =
    tasks.register<AssembleGazeAssets>("assembleGazeAssets") {
        landmarker.set(downloadFaceLandmarker.flatMap { it.output })
        if (bundlePath != null) bundle.set(file(bundlePath))
    }

androidComponents {
    onVariants { variant ->
        variant.sources.assets?.addGeneratedSourceDirectory(assembleGazeAssets, AssembleGazeAssets::outputDir)
    }
}

dependencies {
    implementation(project(":core:domain"))
    implementation(project(":core:ml"))
    implementation(project(":core:camera"))

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.lifecycle.runtime.compose)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(platform(libs.compose.bom))
    implementation(libs.compose.ui)
    implementation(libs.compose.ui.tooling.preview)
    implementation(libs.compose.material3)
    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.hilt.android)
    implementation(libs.hilt.navigation.compose)
    ksp(libs.hilt.compiler)
    debugImplementation(libs.compose.ui.tooling)

    testImplementation(libs.junit4)
    testImplementation(kotlin("test"))
    testImplementation(libs.kotlinx.coroutines.test)
}
