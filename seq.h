#pragma once

#include <vector>
#include <memory>
#include <cudnn.h>

#include "macro.h"
#include "cudnn_enums.h"
#include "cudnn_handles.h"
#include "storage.h"
#include "tensor.h"


struct SeqStruct : Storage {
public:
  using DType = cudnn_enums_auto_export::DType;
public:
  int seq_len;
  std::vector<int> batch_sizes;
  DType dtype;
  std::vector<TensorU> steps;
public:
  std::vector<cudnnTensorDescriptor_t> raw_steps;
public:
  explicit SeqStruct(const std::vector<int> &batch_sizes,
                     int vec_size,
                     const DType &dtype):
    Storage(
      /*ndim=*/3,
      /*dtype=*/dtype,
      /*size=*/0
    ),
    seq_len(batch_sizes.size()),
    batch_sizes(batch_sizes),
    dtype(dtype)
  {
    steps.reserve(seq_len);
    raw_steps.reserve(seq_len);
    for (int batch_size: batch_sizes) {
      TensorU tensor(new TensorStruct(dtype, {batch_size, vec_size, 1}));
      size += tensor->size;
      raw_steps.emplace_back(tensor->desc.get());
      steps.emplace_back(std::move(tensor));
    }
  }
  explicit SeqStruct(int seq_len,
                     int batch_size,
                     int vec_size,
                     const DType &dtype):
    SeqStruct(std::vector<int>(seq_len, batch_size), vec_size, dtype) {
  }
};
using SeqU = std::unique_ptr<SeqStruct>;
using SeqS = std::shared_ptr<SeqStruct>;
