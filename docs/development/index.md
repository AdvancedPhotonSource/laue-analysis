# Development

This page records the current local workflow for Python, native, test, and documentation changes.

```{toctree}
:maxdepth: 1

simulation-performance
```

## Build the project

Install the system dependencies listed in [Installation](../installation.md), then create an editable environment:

```console
$ python -m pip install -e '.[test,docs]'
```

The build invokes the native makefiles and copies their outputs into the source package. Review all build output because the current build script can continue after a native compilation failure.

CUDA is optional. A missing `nvcc` skips the GPU reconstruction build and does not block indexing development.

## Run tests

Run the complete suite from the repository root:

```console
$ python -m pytest
```

Run a focused module while developing a change:

```console
$ python -m pytest tests/test_indexer.py
```

Tests that depend on native indexing skip when `src/laueanalysis/indexing/bin/liblaue.so` is absent. A passing run with skipped native tests does not validate the in-process C path. Review the skip summary.

The `integration` marker identifies tests that require compiled command-line executables. Exclude them when working in an environment without those programs:

```console
$ python -m pytest -m "not integration"
```

## Build the documentation

Install the documentation dependencies and treat warnings as failures:

```console
$ python -m pip install -e '.[docs]'
$ python -m sphinx -E -W --keep-going -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` or serve the directory locally:

```console
$ python -m http.server 8000 --directory docs/_build/html --bind 127.0.0.1
```

## Document public APIs

Use NumPy-style docstrings for public objects. Include a public object explicitly on a curated page under `docs/reference/`. Do not generate references for a whole module because that can expose private helpers and compatibility objects accidentally.

State shapes, dtypes, units, coordinate order, defaults, ownership, exceptions, and observable behavior where relevant. Use cross-references to connect guides and reference entries instead of copying long descriptions.

The documentation writing profile is maintained in `sandbox/writing_format/documentation_style_profile.md` in development workspaces. Apply its scientific notation, coordinate, example, and editing rules to narrative changes.

## Validate examples

Classify examples by what they need:

- Execute pure-Python examples in tests when practical.
- Validate native indexing examples with maintained integration fixtures.
- Keep illustrative snippets short and label synthetic data clearly.
- Do not require native execution during a basic Sphinx HTML build.

A code block that demonstrates exact output must have a test that checks that output. Avoid exact values when the example exists only to show control flow.

## Publish the site

The repository does not yet contain the planned documentation CI and GitHub Pages workflows. Until those workflows are added, build the site locally with the strict command above. Do not commit `docs/_build/`.
