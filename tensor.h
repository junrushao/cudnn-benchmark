#pragma once

#include <vector>
#include <memory>
#include <cudnn.h>

#include "macro.h"
#include "storage.h"

#include "cudnn_handles.h"
#include "cudnn_enums.h"

struct TensorStruct : public Storage {
public:
  using TensorDesc = cudnn_handles_auto_export::TensorDesc;
  using DType = cudnn_enums_auto_export::DType;
public:
  TensorDesc desc{nullptr};
public:
  explicit TensorStruct(const DType &dtype,
                        const std::vector<int> &shape):
    Storage(
      /*ndim=*/shape.size(),
      /*dtype=*/dtype,
      /*size=*/1
    ),
    desc(cudnn_handles_auto_export::CreateTensorDesc())
  {
    int ndim = shape.size();
    std::vector<int> stride(ndim);
    for (int i = ndim - 1; i >= 0; --i) {
      stride[i] = size;
      size *= shape[i];
    }
    CUDNN(SetTensorNdDescriptor(
      /*tensorDesc=*/desc.get(),
      /*dataType=*/dtype.v,
      /*nbDims=*/ndim,
      /*dimA=*/shape.data(),
      /*strideA=*/stride.data())
    );
  }
  ~TensorStruct() {}
};
using TensorU = std::unique_ptr<TensorStruct>;
using TensorS = std::shared_ptr<TensorStruct>;
