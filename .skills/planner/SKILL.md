---
name: planner
description: >-
  Design a Phenocam Vision Edge change as an implementation-ready proposal or
  Markdown specification without modifying production code. Use when a request
  needs explicit scope, structure, invariants, decisions, risks, and validation
  before implementation.
---

<!--
This skill owns requirements and structure planning before approval; it does
not implement, optimize, release, or validate a completed change.
-->

# Planner

Communicate with the user in Italian. Write committed technical specifications,
commands, identifiers, and file names in English.

Obey `AGENTS.md`. The planner may inspect the repository and, only when the user
requests a durable plan, create or update Markdown planning documents. It never
modifies source, tests, configuration, models, dependencies, or release state.

## Right-size the result

- For a small change, provide a concise in-chat structure proposal.
- For a standard change, use one specification with scope, invariants, files,
  tasks, risks, and checks.
- Split a durable plan only when independent milestones genuinely warrant more
  than one file.

Do not create a `plans/` directory or templates by default. Ask a question only
when repository evidence cannot resolve a choice that materially changes
behavior, architecture, dependencies, public interfaces, ownership, or an
irreversible external action.

## Required proposal

Before any implementation, provide:

1. the observable outcome and excluded behavior;
2. the invariant that must remain true;
3. the smallest viable change and principal risk;
4. the complete minimal file tree;
5. one-sentence responsibility and productive-line estimate for every file,
   excluding tests and fixtures;
6. ordered tasks and material dependencies;
7. objective acceptance checks and exact validation commands;
8. decisions needed to prevent implementation drift.

Orchestration files must remain within 200 productive lines. Apply category K
only to the numeric kernels defined by `AGENTS.md`, with the exact marker.
Every file must have one responsibility. Create folders only for domains with
multiple files, and split a domain into a folder rather than prefix-named
siblings.

Wait for explicit user approval or feedback after presenting the structure.
Do not write production code in the same turn as the initial proposal.

## Project questions

Resolve these from code and documentation before asking the user:

- whether a change affects the sixteen-view inference contract;
- whether detection suppression or fixed class configuration changes;
- whether image, metadata, or conditional-deletion transaction ordering changes;
- whether the bundled ONNX model or its strict metadata contract changes;
- whether Raspberry Pi memory, Python version, or release contents change;
- whether release behavior belongs to a frozen tag or ongoing `main`.

Treat all input paths, images, metadata, model tensors, archives, and remote API
responses as untrusted boundaries in the specification.

## Completeness gate

Before presenting the proposal, ensure there are no unresolved material
decisions, vague success terms, uncovered error points, subjective definitions
of done, dangling task dependencies, unsupported external checks, inconsistent
terms, or requirements without implementing tasks. Label unavailable hardware,
network, reference images, or credentials as external verification.

When the user approves, hand the exact approved structure to the implementation
workflow. If implementation exposes an incompatible assumption, update only the
affected proposal and repeat the approval boundary.
