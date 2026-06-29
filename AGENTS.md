# AGENTS.md
on bool3, default env is /tmp/dccvt-venv/bin/python

## Project goal

This repository contains research code for DCCVT, differentiable mesh extraction,
neural SDF experiments, mesh evaluation, and experiment automation.

The priority is correctness, reproducibility, readability, and minimal changes.

## Coding rules

- Prefer simple, explicit Python over clever abstractions.
- Do not introduce large frameworks unless explicitly requested.
- Do not duplicate experiment scripts when a config parameter is enough.
- Do not silently change metrics, seeds, paths, output formats, or dataset filters.
- Keep functions short
- Add type hints for public functions.
- Use dataclasses or typed config objects for experiment parameters.
- Preserve backward compatibility with existing experiment outputs unless told otherwise.

## Experiment rules

Every new experiment must have:
- a config file in `configs/`
- a clear output directory convention
- saved command-line arguments or resolved config
- seed logging
- a short note in `docs/experiments.md`

Never mix baseline logic and proposed-method logic in the same function
unless the distinction is explicit.

## Verification

Before saying a task is complete:
- run the smallest relevant test
- run formatting/linting if configured
- run a smoke test if the code touches experiments
- summarize changed files
- explain any unverified assumptions

# Research Code and Documentation Guidelines

## Core philosophy

This is a research codebase. Prioritize:

1. Correctness
2. Reproducibility
3. Readability
4. Minimality
5. Performance, only after the above are preserved

Code should be simple enough that a future PhD student can understand, modify, and rerun it without needing the original author.

Avoid clever abstractions unless they clearly reduce duplication or make experiments safer to reproduce.

## Research-code style

When writing or modifying code:

* Prefer explicit, readable Python over compact or overly generic code.
* Keep experiment entry points thin.
* Put reusable logic in library modules, not inside experiment scripts.
* Avoid copy-pasting whole experiment scripts to create variants.
* Prefer configuration files or command-line arguments for experiment changes.
* Do not silently change metric definitions, dataset filters, random seeds, output formats, or evaluation logic.
* Do not introduce new dependencies unless necessary.
* Do not hide important research assumptions inside helper functions.
* Use clear names based on research concepts, such as `compute_chamfer_metrics`, `extract_dccvt_mesh`, `load_sdf_samples`, or `run_ablation`.
* Add comments only where they explain non-obvious research logic, numerical assumptions, or implementation tradeoffs.
* Do not over-comment obvious Python syntax.

## Experiment design rules

Every experiment should make clear:

* What method is being tested
* What baseline is being compared
* What config was used
* What command was run
* What input data was used
* What output directory was produced
* What metrics were computed
* What assumptions or limitations apply

New experiment code should be reproducible from a clean command.

## Documentation task standard

When asked to document a subfolder, produce a complete Markdown guide for that subfolder.

The documentation must include:

1. Purpose of the subfolder
2. High-level architecture
3. File-by-file explanation
4. Important classes and functions
5. All command-line entry points
6. All parameters, arguments, and config fields
7. Default values where available
8. Required inputs
9. Generated outputs
10. Example commands
11. Minimal smoke-test command
12. Common failure cases
13. Notes about GPU, CUDA, memory, or dataset requirements
14. Known assumptions and limitations

Do not invent behavior. If a parameter, command, or behavior cannot be verified from the code, mark it as `Unknown` or `Needs verification`.

When documenting parameters, inspect:

* `argparse`
* `click`
* `typer`
* dataclasses
* YAML/JSON config files
* default constants
* README examples
* shell scripts
* training or evaluation entry points

When documenting commands, include:

* Working directory
* Full command
* Required input files
* Expected output files or folders
* Whether the command modifies files
* Whether the command requires GPU
* Whether the command is a smoke test, full experiment, or utility command

The final documentation should be saved as:

`docs/<subfolder_name>_guide.md`

If the subfolder already has documentation, update it instead of creating a duplicate.

## Before editing

For non-trivial tasks, inspect the relevant files first and summarize:

1. What the current code does
2. Which files matter
3. What documentation or code changes are needed
4. What assumptions are uncertain
5. The proposed plan

Do not edit files until the plan is clear.

## Definition of done

A task is complete only when:

* The modified files are listed
* The important behavior is summarized
* Any commands or tests that were run are reported
* Any commands or tests that could not be run are clearly stated
* Any uncertain assumptions are explicitly marked
