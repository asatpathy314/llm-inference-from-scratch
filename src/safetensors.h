#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

class MappedFile {
 public:
  explicit MappedFile(const std::string& path);
  ~MappedFile();

  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  std::span<const uint8_t> bytes() const { return {data_, size_}; }

 private:
  uint8_t* data_ = nullptr;
  std::size_t size_ = 0;
};
