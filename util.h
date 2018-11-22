#pragma once

#include <vector>
#include <assert.h>

#include "macro.h"
#include "cudnn_enums.h"
#include "cudnn_handles.h"
#include "tensor.h"
#include "filter.h"
#include "dropout.h"
#include "seq.h"
#include "rnn.h"
#include "memory_allocator.h"


template <typename T>
__global__ void _fill_kernel(T *array, int size, T value) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int step = gridDim.x * blockDim.x;
  for (int i = idx; i < size; i += step) {
    array[i] = value;
  }
}

template <typename T>
void fill(void *array, int size, T value) {
  T *_array = static_cast<T*>(array);
  const int blockSize = 256;
  const int numBlocks = (size + blockSize - 1) / blockSize;
  _fill_kernel<<<numBlocks, blockSize>>>(_array, size, value);
}

template <typename T>
void copyToDevice(void *dst, n_bytes_t size, const std::vector<T> &src) {
  assert(size == src.size());
  CUDA(Memcpy(dst, src.data(), size * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
void copyToHost(const std::vector<T> &dst, void *src) {
  CUDA(Memcpy((void *)dst.data(), src, dst.size() * sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T>
void print(const std::string &name, const std::vector<int> &shape, const std::vector<T> &src) {
  std::cout << name << " = np.array(";
  bool is_first;
  is_first = true;
  std::cout << "[";
  for (T v : src) {
    if (is_first) {
      is_first = false;
    } else {
      std::cout << ", ";
    }
    std::cout << v;
  }
  std::cout << "], dtype='float32').reshape([";
  is_first = true;
  for (int v : shape) {
    if (is_first) {
      is_first = false;
    } else {
      std::cout << ", ";
    }
    std::cout << v;
  }
  std::cout << "])" << std::endl;
}

template <typename T>
void print(const std::string &name, const std::vector<int> &shape, void *data) {
  size_t size = 1;
  for (int v : shape) {
    size *= v;
  }
  std::vector<T> dst(size);
  copyToHost<T>(dst, data);
  print(name, shape, dst);
}

template <typename T>
std::vector<T> gen_rand_vector(int size) {
  std::vector<T> result(size, 0);
  for (int i = 0; i < size; ++i) {
    result[i] = rand() % 10 - 5;
  }
  return result;
}

template <typename T>
std::vector<T> gen_vector(int size, int value) {
  return std::vector<T>(size, value);
}
