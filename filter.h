#pragma once

#include <vector>
#include <memory>
#include <cudnn.h>

#include "macro.h"
#include "storage.h"

#include "cudnn_handles.h"
#include "cudnn_enums.h"

struct FilterStruct : public Storage {
public:
  using FilterDesc = cudnn_handles_auto_export::FilterDesc;
  using TensorFormat = cudnn_enums_auto_export::TensorFormat;
  using DType = cudnn_enums_auto_export::DType;
public:
  FilterDesc desc{nullptr};
public:
  explicit FilterStruct(const DType &dtype,
                        const TensorFormat &fmt,
                        const std::vector<int> &shape):
    Storage(
      /*ndim=*/shape.size(),
      /*dtype=*/dtype,
      /*size=*/1
    ),
    desc(cudnn_handles_auto_export::CreateFilterDesc())
  {
    CUDNN(SetFilterNdDescriptor(
      /*filterDesc=*/desc.get(),
      /*dataType=*/dtype.v,
      /*format=*/fmt.v,
      /*nbDims=*/ndim,
      /*filterDimA*/shape.data())
    );
    for (int dim : shape) {
      size *= dim;
    }
  }
  ~FilterStruct() {}
};
using FilterU = std::unique_ptr<FilterStruct>;
using FilterS = std::shared_ptr<FilterStruct>;
