# AGENTS.md

## Purpose

This repository should be developed in a simple, direct style: small understandable programs, explicit invariants, few moving parts, and low accidental complexity.

Guided by **KISS** (*Keep It Simple, Stupid*) and the **Unix philosophy**: *"Do one thing and do it well."*

## Core Principles

- Make minimal, focused changes.
- Do not create new files or functions unless requested or clearly beneficial.
- Avoid micro-functions and utility abstractions that protect no invariant.
- Prefer idiomatic, direct, self-contained code and established naming.
- Optimize for understandability first; optimize performance only where it matters.
- Treat unnecessary complexity as a design bug.
- Cut or defer features that add disproportionate complexity.
- Favor designs that can be reasoned about locally.
- Keep interfaces narrow, predictable, and explicit.
- Add inline comments to clarify non-obvious logic, decisions, or trade-offs.

## Cybersecurity

- Write code with cybersecurity in mind.
- Treat all external data (user input, APIs, files) as untrusted; validate, parse, and sanitize strictly before processing.
- Avoid insecure defaults, unsafe input handling, data leaks, injection risks, and unnecessary exposure of sensitive information.
- Fail securely and ensure error messages or logs do not expose system internals, paths, or stack traces.
- Never commit hardcoded secrets, API keys, credentials, or production tokens.
- Minimize attack surface: do not introduce third-party dependencies unless absolutely necessary and thoroughly verified.

## AI Operating Mode

- Do not vibe code.
- Stay within explicit human and repository intent.
- Before changing code, understand existing constraints, invariants, and success criteria.
- Prefer small, well-understood changes over broad rewrites.
- Avoid new abstractions, helpers, frameworks, or dependencies unless they clearly reduce complexity.
- State assumptions, trade-offs, invariants, and risks when relevant.

### Protected AI Development Material

Markdown files used to instruct, coordinate, plan, review, or preserve context
for AI-assisted development are protected repository material.

Never delete, rename, merge, replace, or consolidate `AGENTS.md`, `CLAUDE.md`,
`GEMINI.md`, `.skills/**`, `plans/**`, `DECISIONS.md`, or another Markdown file
explicitly used by an AI development workflow unless the user explicitly
requests that exact operation.

Documentation cleanup must not classify these files as public-documentation
noise. Their wording may be simplified and contradictions may be corrected, but
their entry points, responsibilities, generic skills, references, and historical
development context must remain available.

### Phase 1: Structure Proposal

**Action:** When the user asks for a proposal, plan, or design review, propose the
software structure according to the following strict constraints:

- **File and Directory Tree:** Include only what is strictly necessary for the approved solution.
- **Self-Explanatory Names:** Use clear, direct nouns (e.g., `console`, `runtime`, `parser`). Do not use abstract or verb-based names.
- **Line Count Estimation:** For each file, provide an explicit estimation of productive lines of code, excluding tests and test fixtures (e.g., `parser.py (~120 lines, excluding tests)`).
- **200-Line Limit — orchestration code:** Code that dispatches work, owns resources, performs configuration or budget arithmetic, handles I/O, or wires the inference pipeline is limited to **200 productive lines**. Split an estimate above 200 before implementation, not after the file has grown. Tests and test fixtures never count toward this limit.
- **No line limit — category K:** A single dense numeric kernel, or a cohesive family of format variants of one numeric operation, has no line limit when it contains no cross-operation dispatch, I/O, or resource ownership. Declare it once with `# AGENTS deroga K: <note>`. The category is narrow, never a blanket for inference or NumPy files.
- **Single Responsibility:** Each file must do one thing and one thing only. If its purpose cannot be described in a single sentence, it must be split.
- **Folders Only When Warranted:** Introduce a directory only when its domain genuinely splits into more than one file. A domain that fits in one file stays a clear domain-named file, never a folder containing only `__init__.py` and one implementation file.
- **Split by folder, not by prefix:** When a file is split, group the parts in a domain folder (`domain/{__init__.py, part.py, ...}`), never as prefix-named siblings (`domain.py` plus `domain_part.py`). A distinct helper may remain a sibling when it belongs to a different existing responsibility.

> **Workflow Boundary:** Wait only when the user requested a structure proposal,
> plan, or design review without authorizing implementation. An explicit request
> to implement, fix, optimize, benchmark, investigate, or execute an existing
> specification is itself approval to enter Phase 2 within that request's scope.
> Do not ask for duplicate approval or per-step confirmation.

If implementation reveals an in-scope structural change that was not knowable
up front, record its tree and productive-line estimates before making it, then
continue when it is local, reversible, and verifiable. Ask only when the choice
expands scope, changes a public contract or dependency, has an irreversible or
external effect, or cannot be resolved safely from evidence.

#### Autonomous Performance Investigations

An explicit performance request authorizes the local cycle of baseline,
instrumentation, profiling, hypothesis, reversible experiment, correctness
check, measurement, retention or restoration, and renewed profiling. Follow the
measured bottleneck across image loading, view generation, ONNX Runtime,
detection merging and suppression, rendering, metadata, filesystem I/O, and
test or benchmark support.

Keep each candidate isolated and reproducible. Fix the model, inputs, output
mode, Python environment, thread count, hardware, and thermal conditions between
measurements. Temporary instrumentation must be removed before completion unless
the user requested it as a maintained interface.

Do not request approval between in-scope candidates. Retain only changes that
pass correctness gates and improve the declared metric beyond observed noise.
This autonomy does not authorize pushes, pull requests, merges, secrets,
dependencies, public API changes, destructive operations, or unrelated work.

### Phase 2: Implementation Phase

Once implementation is explicitly requested or a proposed structure is
approved, ensure that every file adheres to the following checklist:

- [ ] **Architectural comment where needed:** Explain purpose and context for orchestration, resource ownership, category-K exemptions, and non-obvious security or format boundaries. Keep simple files self-explanatory.
- [ ] **Compliant implementation:** Stay within the declared productive-line estimate, counting productive code and non-test comments.
- [ ] **Inline invariants:** Documented and explained at critical logical and state-mutation points.
- [ ] **Minimal tests:** Attached, implemented, or described contextually within or alongside the file block.

## Design Rules

- Start from the most direct working model and make invalid states hard to express.
- Use dedicated types, modules, and restricted visibility only where they protect invariants.
- Make ownership, mutation, side effects, dependencies, and error boundaries explicit.
- Prefer simple data flow and clear responsibility boundaries.
- Use shared state, concurrency, metaprogramming, decorators, or dynamic dispatch only for a concrete boundary.
- Prefer safe, idiomatic language features; use low-level, unsafe, reflective, or escape-hatch mechanisms only when necessary, narrow, documented, and backed by clear invariants.

## Performance

- Do not optimize blindly.
- First choose a design that is small, correct, and inspectable.
- Keep the bundled ONNX model, representative input, output mode, thread count, and target hardware fixed while comparing candidates.
- Optimize only after identifying a real bottleneck.
- Prefer structural performance improvements over micro-optimizations.
- Stop when the explicit completion conditions are satisfied; do not optimize beyond a sufficient measured result.

## Change Policy

Before editing, identify:
- The invariant that must remain true;
- The smallest viable change;
- The main risk introduced.

After editing, verify whether the change reduced or increased conceptual complexity. If complexity increased, explain why it was unavoidable. If a feature request conflicts with simplicity, prefer the smallest version that still satisfies the explicitly authorized request.

## Release Versioning

- Keep the software version in one runtime source: `phenocam/__init__.py`,
  with exactly one literal declaration `__version__ = "MAJOR.MINOR.PATCH"`.
- Before committing a release, update that declaration to the intended version.
  Detection metadata must use it; do not add another runtime version constant
  or require Git at runtime.
- Keep software and model versions independent. A software release alone does
  not justify changing `model_version` or the model artifacts.
- Keep the bundled model identity in `models/yolo26n-phenocam.json`:
  `model_id` is `yolo26n-phenocam` and `model_version` is `0.1.6` for the
  historical v6 artifact. In this experimental model series, revision vN uses
  `0.1.N`; this is a project convention, not the software release number or
  the Ultralytics exporter version. Change the series only by explicit decision.
- Keep the stable model filenames independent of the revision. Record a new
  version and matching hashes for a new model artifact; never assign the same
  model identity/version to different ONNX bytes. Preserve historical provenance
  and acceptance status. This identity correction does not change model bytes.
- Detection metadata must obtain model identity from the selected ONNX file's
  sibling JSON receipt, after validating its fields and `onnx_sha256` against
  the selected file. Do not duplicate identity constants in runtime code or
  infer versions from filenames. An absent receipt yields `model_id=unknown`
  and the conventional initial `model_version=0.1.0` (not a verified revision); an invalid
  or mismatched receipt with `--meta` must fail before output writes or deletion.
- Update current installation commands, metadata examples, and release-related
  test expectations together. Preserve historical release references.
- For commits declaring a version, the declaration, packaging argument, archive
  name, and release tag (`vMAJOR.MINOR.PATCH`) must agree. Preserve packaging
  compatibility for historical commits without the declaration.
- Package the exact commit selected for release using `scripts/package.py`;
  never bypass its version check or execute source code to read its version.
- Before publication, require passing CI for that commit, verify the archive
  and checksum, and check that its code writes the intended `software_version`,
  `model_id`, and `model_version` to a temporary `.meta` file. Verify the packaged
  model bytes against the receipt; a model version update alone does not prove
  a quality improvement or authorize promotion out of experimental status.
- Publish corrections as a new patch release. Do not move existing release tags
  or replace published assets. Preserve experimental/prerelease status unless
  promotion is explicitly authorized and supported by validation.

## Non-Goals

- No fashionable abstractions for their own sake.
- No premature generality.
- No unnecessary indirection.
- No broad rewrites without clear structural benefit.
- No code the AI cannot explain line by line.

---

Once changes are complete, update the documentation as needed.
