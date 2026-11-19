# ADR-0008: Interaction: hysteresis cell selection + voluntary-blink confirm

- **Status:** Accepted
- **Date:** 2026-11-19

## Context

The use case is: look at a cell to select it, blink to confirm. Natural blinks (~100–300 ms) must not confirm, and gaze is unreliable while the eyelid closes.

## Decision

Hover cell changes only after the gaze stays in a new cell for ~250 ms. Confirmation is a deliberate long blink (both eyes closed 400–1500 ms, blendshape scores with hysteresis and per-user thresholds); double-blink is an optional alternative. The cell is locked at blink onset from pre-closure frames. 500 ms cooldown after a selection.

## Consequences

Low false-confirm rate by construction; measured in E5 (false confirms per minute, selection accuracy and time per grid size).
