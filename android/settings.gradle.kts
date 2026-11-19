pluginManagement {
    repositories {
        gradlePluginPortal()
        mavenCentral()
        google()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        mavenCentral()
        google()
    }
}

rootProject.name = "eyegaze-android"

include(":core:domain") // pure Kotlin/JVM: builds and tests without the Android SDK

// Android modules need the SDK; without it, :core:domain (all domain logic + golden parity tests) still builds.
val localProperties = file("local.properties")
val androidSdkAvailable =
    listOf("ANDROID_HOME", "ANDROID_SDK_ROOT").any { !System.getenv(it).isNullOrBlank() } ||
        (localProperties.isFile && localProperties.readLines().any { it.trim().startsWith("sdk.dir=") })
if (androidSdkAvailable) {
    include(":core:ml", ":core:camera", ":app")
} else {
    logger.lifecycle("Android SDK not found: configuring only :core:domain")
}
