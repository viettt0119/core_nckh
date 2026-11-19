import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "app.eyegaze.ml"
    compileSdk = 35

    defaultConfig {
        minSdk = 26
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    androidResources {
        noCompress += listOf("tflite", "task")
    }
    sourceSets {
        // Instrumented golden tests read the same fixtures the JVM and Python tests use.
        getByName("androidTest").assets.srcDir(rootProject.layout.projectDirectory.dir("../contracts"))
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_17)
        allWarningsAsErrors.set(true)
    }
    explicitApi()
}

dependencies {
    api(project(":core:domain"))
    implementation(libs.litert)
    implementation(libs.litert.gpu)
    implementation(libs.litert.gpu.api)
    implementation(libs.mediapipe.tasks.vision)

    testImplementation(libs.junit4)
    testImplementation(kotlin("test"))
    androidTestImplementation(libs.androidx.test.junit)
    androidTestImplementation(libs.androidx.test.runner)
    androidTestImplementation(libs.kotlinx.serialization.json)
    androidTestImplementation(kotlin("test"))
}
