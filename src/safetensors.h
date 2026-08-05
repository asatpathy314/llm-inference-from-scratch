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

enum class DType {
  I8,
  I16,
  I32,
  I64,
  BF16,
  F16,
  F32,
  F64,
  U8,
  BOOL,
};

struct TensorView {
  DType dtype;
  std::vector<int64_t> shape;
  std::span<const uint8_t> data;

  int64_t numel() const;
};

using TensorMap = std::unordered_map<std::string, TensorView>;
TensorMap parse_header(const std::span<const uint8_t> data);

class SafeTensors {
 public:
  explicit SafeTensors(const std::string& path);

  const TensorView& get(const std::string& name) const;
  const std::unordered_map<std::string, TensorView>& tensors() const { return tensors_; }

 private:
  MappedFile file_;
  std::unordered_map<std::string, TensorView> tensors_;
};
