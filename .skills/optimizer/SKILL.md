---
name: optimizer
description: >-
  Optimize implemented Phenocam Vision Edge code while preserving behavior and
  public interfaces. Use for scoped latency, memory, I/O, or algorithmic work
  that can be justified by measurements or a clear complexity reduction.
---

<!--
This skill owns conservative optimization of implemented code; feature design,
broad cleanup, and unmeasured tuning remain outside its scope.
-->

# Optimizer

Communicate with the user in Italian. Keep code, commands, identifiers, and
committed technical documentation in English.

Obey `AGENTS.md`. Record the smallest structure before editing; an explicit
optimization request authorizes the local measure-change-verify cycle. Preserve
CLI behavior, detections, output products, metadata, error semantics, filesystem
ordering, dependencies, and supported platforms.

## Scope and evidence

Use the explicit scope. Otherwise inspect only code changed on the current
branch plus uncommitted changes. Nearby code may be read to understand direct
call paths but remains out of scope.

An optimization may be applied only when both conditions hold:

1. relevant tests or a declared correctness oracle prove behavior preservation;
2. repeated measurement shows a gain beyond observed variance, or a direct
   complexity argument proves less work on the exercised path.

Without both legs, report the candidate and leave production code unchanged.

## Classification

Classify every finding once:

- **Optimization:** local, proven safe, and supported by both evidence legs;
- **Correctness risk:** a resource problem that can also cause wrong behavior;
- **Speculative:** gain or equivalence is not demonstrated;
- **Out of scope:** outside the approved area.

Only approved Optimizations and clearly local correctness risks may be changed.
Never remove validation, logging, or error handling for speed. Do not introduce
dependencies, global caches, concurrency, model changes, lower precision, or
new batch lifetime semantics without explicit design approval.

## Targets

Inspect measured paths for repeated NumPy allocation or conversion, avoidable
image copies, redundant parsing, repeated filesystem work, unnecessary sorting
or scans, poorly bounded intermediate arrays, and work repeated across the
sixteen views. Treat ONNX Runtime execution as an external optimized boundary;
do not guess at provider or thread changes.

The `scripts/batch.sh` one-process-per-image model intentionally isolates
failures. Reusing a session across images is an architectural and behavioral
decision, not a local optimization.

## Workflow

1. Record the current commit, worktree scope, representative input, model SHA,
   Python and dependency versions, thread count, hardware, and baseline.
2. Attribute the measured cost before selecting a candidate.
3. Record the invariant, smallest diff, risk, expected evidence, file tree, and
   estimates before the first production edit.
4. Change one independent factor.
5. Run correctness checks before repeating the identical measurement.
6. Retain the candidate only if both evidence legs pass; otherwise restore only
   that candidate and record the negative result.

Try no more than three focused variants for one hypothesis. Stop when no
measured candidates remain or the requested budget is exhausted. Do not commit
or push unless the user requested it.

## Verification and report

Run focused tests, the full unit suite, shell syntax checks, and
`git diff --check`. Real-image tests that skip because inputs are absent are an
external limitation. Report files inspected and changed, before/after evidence,
preserved behavior, speculative candidates, correctness risks, all validation
results, and one English commit-title suggestion when changes remain.
