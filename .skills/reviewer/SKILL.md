---
name: reviewer
description: >-
  Independently verify the complete current Phenocam Vision Edge change without
  modifying it. Use as the final read-only step for fresh validation, a concise
  user summary, and an OK, REVIEW, or PROBLEMS verdict.
---

<!--
This skill owns final read-only verification and the user-facing verdict;
planning, fixes, optimization, commits, and publication remain separate.
-->

# Reviewer

Communicate with the user in Italian. This skill is strictly read-only: never
edit files, fix findings, commit, tag, publish, or run destructive commands.

## Scope

Use the user's explicit scope. Otherwise review the union of the branch diff
against its merge base with `main`, staged changes, unstaged changes, and
untracked files. Read commit messages, `AGENTS.md`, the approved proposal or
specification when available, and durable decision records. Treat prior agent
reports only as hints.

If no approved specification exists, verify internal consistency and observable
tests but do not invent intended product behavior.

## Review

Inspect the complete scoped diff for:

- missed requirements or behavior outside the approved scope;
- violations of file responsibility, productive-line estimates, or the
  200-line orchestration limit;
- changes to the sixteen-view inference and suppression contracts;
- unsafe path, symlink, metadata, deletion, archive, or external-input handling;
- accidental model, dependency, generated artifact, secret, or personal path;
- documentation that describes `main` as though it were the frozen release;
- skipped work, questionable decisions, or tests weakened to accept a change.

## Fresh verification

Run the narrowest named tests and then the applicable full gates:

```sh
.venv/bin/python -m unittest discover -s tests -v
sh -n scripts/batch.sh
sh -n scripts/installer.sh
git diff --check
```

For skill changes, run the skill validator on each changed skill folder. For
packaging changes, run `tests/test_release_package.py` explicitly. Record every
command as pass, fail, or unavailable. A skipped real-image test is not evidence
of Raspberry Pi inference correctness.

## Output

Keep the result short and group it by user-visible meaning:

```text
## Sintesi

**Cosa è cambiato**
- ...

**Verifica**
- ...

**Da controllare**
- ...

**Verdetto: OK | REVIEW | PROBLEMS**
Una frase con il motivo.
```

Omit empty sections except the verdict. Use **OK** only when all applicable
checks pass and no unresolved risk remains; **REVIEW** when checks pass but a
decision, skipped external check, or limitation needs attention; **PROBLEMS**
for failed validation, incomplete work, or a visible regression. Always produce
a verdict even when a tool or external resource is unavailable.
