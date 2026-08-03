#include "safetensors.h"

#include <fcntl.h>     // open
#include <sys/mman.h>  // mmap, munmap
#include <sys/stat.h>  // fstat
#include <sys/types.h>
#include <unistd.h>  // close

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
