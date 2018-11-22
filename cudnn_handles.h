#pragma once

#include <memory>
#include <cudnn.h>
#include "macro.h"

namespace cudnn_handles_auto_export {

// Hack around to fix irregular naming
#define cudnnContextStruct cudnnContext
#define cudnnCreateContextDescriptor cudnnCreate
#define cudnnDestroyContextDescriptor cudnnDestroy
#define DEFINE_CUDNN_POD(type_name, cudnn_type_name)                                      \
  struct type_name##Deleter {                                                             \
    void operator () (cudnn##cudnn_type_name##Struct *ptr) const {                        \
      CUDNN(Destroy##cudnn_type_name##Descriptor(ptr));                                   \
    };                                                                                    \
  };                                                                                      \
  using type_name = std::unique_ptr<cudnn##cudnn_type_name##Struct, type_name##Deleter>;  \
  type_name Create##type_name () {                                                        \
    cudnn##cudnn_type_name##Struct *ptr;                                                  \
    CUDNN(Create##cudnn_type_name##Descriptor(&ptr));                                     \
    return type_name(ptr);                                                                \
  }

DEFINE_CUDNN_POD(Context, Context);
DEFINE_CUDNN_POD(TensorDesc, Tensor);
DEFINE_CUDNN_POD(FilterDesc, Filter);
DEFINE_CUDNN_POD(DropoutDesc, Dropout);
DEFINE_CUDNN_POD(RNNDesc, RNN);

#undef DEFINE_CUDNN_POD
#undef cudnnDestroyContextDescriptor
#undef cudnnCreateContextDescriptor
#undef cudnnContextStruct

}  // namespace cudnn_handles_auto_export
