# ADR-0009: Android: Clean Architecture, multi-module, offline-only

- **Status:** Accepted
- **Date:** 2026-11-19

## Context

The app must be testable, maintainable and demonstrably private.

## Decision

Kotlin, Jetpack Compose, Hilt, Coroutines/Flow, CameraX, Room, DataStore. Modules: `:core:domain` (pure Kotlin/JVM: preprocessing, filters, blink FSM, selector, ridge), `:core:ml`, `:core:camera`, `:core:data`, `:core:ui`, `:feature:*`, `:app`. Dependencies point inward to domain ports. No `INTERNET` permission. minSdk 26.

## Consequences

Domain logic runs in fast JVM unit tests and golden parity tests. Privacy is guaranteed by the manifest, not by policy. Network-oriented resilience patterns (retry, circuit breaker) do not apply; we use frame backpressure, GPU→CPU delegate fallback and fail-fast model checks instead.
