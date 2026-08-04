#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

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

struct TensorView {
  std::string dtype;
  std::vector<int64_t> shape;
  std::span<const uint8_t> data;
};

class SafeTensors {
 public:
  explicit SafeTensors(const std::string& path);

  const TensorView& get(const std::string& name) const;
  const std::unordered_map<std::string, TensorView>& tensors() const { return tensors_; }

 private:
  MappedFile file_;
  std::unordered_map<std::string, TensorView> tensors_;
};
