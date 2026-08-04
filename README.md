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

## C++ side

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
./build/inference "$(find ~/.cache/huggingface -name model.safetensors | head -1)"
```

The configure step is only needed after `CMakeLists.txt` changes; day to day just
run the build step. Debug builds carry ASan and UBSan, so they are slow on
purpose. When benchmarking, strip the sanitizer flags from `CMakeLists.txt` before measuring
anything.
