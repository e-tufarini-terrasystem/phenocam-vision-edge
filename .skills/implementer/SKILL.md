---
name: implementer
description: >-
  Implement an explicitly requested or approved Phenocam Vision Edge change
  with a minimal, verified diff. Use for authorized implementation or execution
  of an existing specification; do not use for planning-only work.
---

<!--
This skill owns approved implementation and proportional verification;
requirements planning, release publication, and final read-only review remain
separate responsibilities.
-->

# Implementer

Communicate with the user in Italian. Keep code, identifiers, comments, commit
titles, and committed technical documentation in English.

Implement only the authorized outcome. Obey `AGENTS.md`; an explicit request to
implement is sufficient to enter Phase 2, and its checklist applies to every
touched file.

## Input and boundary

An explicit implementation request, approved in-chat structure, or approved
Markdown specification is enough.
Extract the observable outcome, in-scope files, excluded behavior, invariants,
risks, line estimates, and verification commands. Do not request duplicate
approval unless a newly discovered choice would change behavior, architecture,
dependencies, public interfaces, data ownership, or external state.

Never infer permission to publish releases, push, deploy, rewrite Git history,
delete unrelated data, install system packages, or call side-effecting external
services. Preserve all unrelated worktree changes.

## Project invariants

Apply only the invariants relevant to the change, including:

- inference uses the bundled end-to-end `models/yolo26n.onnx` graph;
- one image is processed sequentially as one full view plus fifteen crops;
- model metadata and tensor shapes are validated before inference;
- enabled-class selection occurs after global detection suppression;
- requested images precede metadata update, which precedes conditional input
  deletion;
- failures expose stable user-facing errors without host details;
- the release archive contains only the explicit runtime allowlist.

Do not weaken input validation, output-path identity checks, atomic metadata
replacement, or release path safety to make a change easier.

## Workflow

### Inspect

Read `AGENTS.md`, the approved structure, affected implementation, direct
callers, adjacent tests, documentation, and the current worktree. Confirm the
approved productive-line estimates still fit.

### Implement

Work in small coherent edits. Prefer existing types and direct control flow over
new helpers or dependencies. Add the required architectural comment to every
new file and explain non-obvious invariants at state-mutation points. Do not
perform opportunistic cleanup.

If reality requires an unforeseen local, reversible, and verifiable structural
change, record the revised tree, estimates, reason, and risk before continuing.
Ask only when the choice expands scope, behavior, architecture, dependencies,
public interfaces, ownership, or external state.

### Verify

Run focused tests first and then, when applicable:

```sh
.venv/bin/python -m unittest discover -s tests -v
sh -n scripts/batch.sh
sh -n scripts/installer.sh
git diff --check
```

Run `scripts/export/fp32.py` only when model export is in scope and its separate
export environment is already available. Treat missing Raspberry Pi hardware,
reference images, or export dependencies as external verification, not success.
Do not install them merely to complete a local check.

For changed skills, run the repository-independent skill validator against each
changed skill directory. For release packaging changes, run
`tests/test_release_package.py` explicitly before the full suite.

### Self-review

Inspect the final diff for scope drift, behavioral changes, secrets, debug code,
unnecessary abstractions, documentation drift, single responsibility, and the
200-productive-line limit. State whether conceptual complexity decreased or why
any increase was unavoidable.

## Final report

Report the outcome, preserved invariants, material files changed, every command
and result, external verification still required, and remaining limitations.
When files changed, suggest one concise English commit title matching repository
conventions. Never claim a check that did not run.
