import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlin.jvm)
    alias(libs.plugins.kotlin.serialization)
    alias(libs.plugins.ktlint)
}

// Pure Kotlin/JVM: no Android dependency, so the whole domain is tested with fast JVM unit tests.
// Bytecode target 17 keeps it consumable by Android modules.
java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_17)
        allWarningsAsErrors.set(true)
        freeCompilerArgs.add("-Xjsr305=strict")
    }
    explicitApi()
}

dependencies {
    implementation(libs.kotlinx.serialization.json)

    testImplementation(kotlin("test"))
    testImplementation(platform(libs.junit.bom))
    testImplementation(libs.junit.jupiter)
    testRuntimeOnly(libs.junit.platform.launcher)
}

val contractsDir = rootProject.layout.projectDirectory.dir("../contracts")

tasks.test {
    useJUnitPlatform()
    // Golden fixtures are produced by the Python reference implementation (research/).
    inputs.dir(contractsDir).withPropertyName("contracts").withPathSensitivity(PathSensitivity.RELATIVE)
    systemProperty("eyegaze.contracts", contractsDir.asFile.absolutePath)
    testLogging {
        events("failed")
        exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
    }
}

ktlint {
    version.set("1.5.0")
}
