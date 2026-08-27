---
name: deep-clean
description: >-
  Simplify an existing Phenocam Vision Edge change without altering behavior.
  Use for thorough removal of dead Python or shell code, duplication, stale
  comments, unnecessary indirection, or accidental complexity in an explicitly
  requested scope, including an exhaustive repository pass.
---

<!--
This skill owns behavior-preserving cleanup in the requested repository scope;
feature work, redesign, and performance experiments remain separate.
-->

# Deep Clean

Communicate with the user in Italian. Keep code, commands, identifiers, and
committed technical documentation in English.

Obey `AGENTS.md`. Cleanup must reduce conceptual complexity without changing
CLI behavior, detection results, output images, metadata, deletion ordering,
release contents, dependencies, or supported runtime requirements.

## Scope

Use the scope named by the user. If the request refers to the current change,
limit inspection to the branch diff and uncommitted changes. Read the affected
files, their direct consumers, and their tests before proposing edits.

Before changing files, record:

1. the behavior and invariants that must remain true;
2. the smallest complete cleanup;
3. the principal regression risk;
4. the required structure and line estimates from `AGENTS.md`.

An explicit cleanup request authorizes local Phase 2 edits within that scope.
Wait only when the user requested a proposal without authorizing implementation,
or when a choice expands behavior, architecture, dependencies, or external state.

## Valid candidates

- private Python code with no runtime, test, export, or packaging consumer;
- repeated logic replaceable by an existing direct form;
- wrappers or conversions that protect no invariant;
- obsolete or contradictory comments;
- unused imports, variables, shell branches, and configuration;
- equivalent branches that can be unified without changing evaluation order,
  errors, filesystem effects, or output ordering.

Search all call sites before declaring code dead. Account for `python -m
phenocam`, tests and mocks, shell entrypoints, model export, release allowlists,
and optional real-image tests. A path is not dead merely because the current
host is not a Raspberry Pi or lacks the untracked reference images.

## Cleanup loop

For each independent candidate:

1. explain the complexity being removed;
2. apply the smallest focused diff;
3. avoid unrelated formatting, renaming, and optimization;
4. run the narrowest relevant test, then the applicable repository gates;
5. retain the change only when behavior remains demonstrated;
6. otherwise restore only that candidate and record why it is load-bearing.

Do not add dependencies, generic helpers, or future-facing structure. If the
claimed benefit requires timing data, leave it to an optimization workflow.

## Verification

Run the complete unit suite when runtime dependencies are available:

```sh
.venv/bin/python -m unittest discover -s tests -v
sh -n scripts/batch.sh
sh -n scripts/installer.sh
git diff --check
```

Report unavailable external checks, including Raspberry Pi execution or absent
reference images, without presenting a skipped check as proof. The final report
lists simplifications retained, rejected candidates, commands run, and results.
