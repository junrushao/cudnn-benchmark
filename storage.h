#pragma once

#include "cudnn_enums.h"

struct Storage {
public:
  using DType = cudnn_enums_auto_export::DType;
public:
  n_dim_t ndim{0};
  n_bytes_t size{0};
  DType dtype{DType::kFloat};
  void *data{nullptr}; // no ownership
public:
  explicit Storage(n_dim_t ndim, DType dtype, n_bytes_t size):
    ndim(ndim),
    dtype(dtype),
    size(size),
    data(nullptr)
  {
  }
};
