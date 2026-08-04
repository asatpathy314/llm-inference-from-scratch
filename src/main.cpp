#include <cstdio>

#include "safetensors.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <model.safetensors>\n", argv[0]);
    return 1;
  }

  MappedFile model(argv[1]);
  std::span<const uint8_t> file_span = model.bytes();

  std::printf("model path: %s\n", argv[1]);
  std::printf("model size: %zu\n", file_span.size());
  return 0;
}
