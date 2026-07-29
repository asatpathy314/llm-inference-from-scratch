"""
Exploring tensors in Llama-3.2.

Example of tensor data information in the header.
  "model.layers.0.input_layernorm.weight": {
    "dtype": "BF16",
    "shape": [
      2048
    ],
    "data_offsets": [
      525336576,
      525340672
    ]
"""

import json
import math
import os
from itertools import pairwise
from pathlib import Path

from huggingface_hub import hf_hub_download

MODEL = "meta-llama/Llama-3.2-1B"
OUTDIR = Path(__file__).resolve().parent.parent / "ref"
HEAD_DIM = 64
DTYPE_TO_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "U8": 1,
    "BOOL": 1,
}


def inspect():
    path = hf_hub_download(MODEL, "model.safetensors")

    # this isn't strictly necessary, but is useful for later C++ development
    with open(path, "rb") as model_bytes:
        # the first 8 bytes of the file contain the length of the header
        header_len = int.from_bytes(model_bytes.read(8), "little")

        # notably the "data_offsets" attribute is relative to 8 + header_len
        header: dict = json.loads(model_bytes.read(header_len))
        data_start = 8 + header_len

        # remove the "metadata" key -- optional in the spec, so pop with a default
        header.pop("__metadata__", None)

        # size check: a tensor's byte span is exactly prod(shape) * sizeof(dtype).
        # This is what proves the blob is contiguous row-major with no padding.
        spans = []
        for name, info in header.items():
            starting_pos, ending_pos = info["data_offsets"]
            memory_size = ending_pos - starting_pos
            expected_size = math.prod(info["shape"]) * DTYPE_TO_BYTES[info["dtype"]]
            assert memory_size == expected_size, (
                f"{name}: {memory_size} bytes, shape implies {expected_size}"
            )
            spans.append((starting_pos, ending_pos, name))

        # tiling check: in offset order, each tensor starts exactly where the last ended
        spans.sort()
        for (_, prev_end, prev_name), (next_start, _, next_name) in pairwise(spans):
            assert prev_end == next_start, f"gap between {prev_name} and {next_name}"

        # file size check: the final tensor ends exactly at EOF
        file_size = os.path.getsize(path)
        blob_end = data_start + spans[-1][1]
        assert blob_end == file_size, f"blob ends at {blob_end}, file is {file_size}"

        # summary
        print(
            f"Total parameter count is {sum(math.prod(info['shape']) for info in header.values())}."
        )
        print(f"Blob size is {file_size - data_start}.")

        # print model tensor information
        for name, info in header.items():
            if ".layers" in name and not name.startswith("model.layers.0."):
                continue
            print(name, info["dtype"], info["shape"])

        # calculate the number of attention heads
        q_proj = "model.layers.0.self_attn.q_proj.weight"
        k_proj = "model.layers.0.self_attn.k_proj.weight"
        n_q_heads = header[q_proj]["shape"][0] // HEAD_DIM
        n_kv_heads = header[k_proj]["shape"][0] // HEAD_DIM
        assert n_q_heads == 32, f"expected 32 query heads, got {n_q_heads}"
        assert n_kv_heads == 8, f"expected 8 kv heads, got {n_kv_heads}"

        # tied embeddings: the output projection reuses model.embed_tokens.weight
        assert "lm_head.weight" not in header, "header contains lm_head.weight"


if __name__ == "__main__":
    inspect()
