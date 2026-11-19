# ADR-0001: Monorepo: research/, android/, contracts/

- **Status:** Accepted
- **Date:** 2026-11-19

## Context

The legacy repo held three diverging implementations of the same pipeline (src/, Colab1.1/, core_ai/cpp). The Colab copy returned a dict where the main code returned a tuple; the C++ core transposed an already-HWC tensor. Every bug traced back to logic existing more than once.

## Decision

One repository with three top-level parts: `research/` (Python: data, training, evaluation, export), `android/` (Kotlin app), `contracts/` (language-neutral specs and golden fixtures that bind the two). The C++ core, the Colab copy and the loose root scripts are deleted; the Colab notebook becomes a thin wrapper that clones the repo and calls the CLI.

## Consequences

Single source of truth per concern. Cross-language agreement is enforced by tests against `contracts/`, not by convention. Legacy code remains in git history (commit `db82694`).
