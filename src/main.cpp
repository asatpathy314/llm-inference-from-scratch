#include <cstdio>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <model.safetensors>\n", argv[0]);
    return 1;
  }
  std::printf("model path: %s\n", argv[1]);
  return 0;
}
