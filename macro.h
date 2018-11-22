#pragma once

#include <sstream>
#include "cuda.h"

#define NO_COPY_MOVE_ASSIGN(T)                \
        T(T const &) = delete;                \
        void operator=(T const &t) = delete;  \
        T(T &&) = delete

#define CUDA(call) {                                        \
  cudaError_t status = cuda ## call;                        \
  if (status != cudaSuccess) {                              \
    std::ostringstream os;                                  \
    os << __FILE__ << ":"                                   \
       << __LINE__ << ": CUDA Error [" << status << "] "    \
       << cudaGetErrorString(status);                       \
    throw std::runtime_error(os.str());                     \
  }                                                         \
}

#define CUDNN(call) {                                       \
  cudnnStatus_t status = cudnn ## call;                     \
  if (status != CUDNN_STATUS_SUCCESS) {                     \
    std::ostringstream os;                                  \
    os << __FILE__ << ":"                                   \
       << __LINE__ << ": CUDNN Error [" << status << "] "   \
       << cudnnGetErrorString(status);                      \
    throw std::runtime_error(os.str());                     \
  }                                                         \
}

using n_dim_t = size_t;
using n_bytes_t = size_t;
using seed_t = unsigned long long;
