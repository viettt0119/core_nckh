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

// Pure Kotlin/JVM modules build anywhere. Android modules (M6+) need the Android SDK and are added here then.
include(":core:domain")
