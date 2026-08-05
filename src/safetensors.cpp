#include "safetensors.h"

#include <fcntl.h>     // open
#include <sys/mman.h>  // mmap, munmap
#include <sys/stat.h>  // fstat
#include <sys/types.h>
#include <unistd.h>  // close

#include <cstring>   // memcpy
#include <iostream>  // cout
#include <nlohmann/json.hpp>
#include <numeric>  // accumulate
#include <stdexcept>
#include <system_error>

MappedFile::MappedFile(const std::string& path) {
  int fd = open(path.c_str(), O_RDONLY);
  if (fd == -1) {
    throw std::system_error(errno, std::generic_category(), "open failed: " + path);
  }

  struct stat st;
  if (fstat(fd, &st) == -1) {
    close(fd);
    throw std::system_error(errno, std::generic_category(), "fstat failed: " + path);
  }

  void* mmap_ptr = static_cast<uint8_t*>(mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0));
  close(fd);
  if (mmap_ptr == MAP_FAILED)
    throw std::system_error(errno, std::generic_category(), "mmap failed: " + path);

  data_ = static_cast<uint8_t*>(mmap_ptr);
  size_ = st.st_size;
}

MappedFile::~MappedFile() { ::munmap(data_, size_); }

SafeTensors::SafeTensors(const std::string& path) : file_(path) {
  std::span<const uint8_t> span = file_.bytes();

  // first 8 bytes are the header length
  uint64_t header_len = 0;
  std::memcpy(&header_len, span.data(), sizeof(header_len));
  nlohmann::json header = nlohmann::json::parse(span.subspan(8, header_len));

  std::cout << header_len << std::endl;
}

int64_t TensorView::numel() const {
  return std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
}
