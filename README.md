# llm-inference-from-scratch

A Llama-3.2-1B inference engine written from scratch in C++/CUDA.

## Layout

```
tools/    Python scripts for development.
src/      C++/CUDA.                                      (not yet)
tests/    Correctness checks against ref/.               (not yet)
ref/      Generated fp32 reference tensors + meta.json.  Not committed.
build/    CMake output.                                  Not committed.
```

## Python side

```sh
uv sync
uv run tools/00_dump_reference.py
```

Llama-3.2 is a gated repo, so authenticate once first:

```sh
uvx hf auth login
```
