#pragma once

#include <cudnn.h>
#include "./util.h"

struct Dropout {
public:
  cudnnDropoutDescriptor_t desc = nullptr;
  size_t state_size = 0;
  void *states = nullptr;
public:
  explicit Dropout(const DnnHandle &handle,
                   float dropout = 0.0,
                   unsigned long long seed = 0) {
    // this->desc
    CUDNN(CreateDropoutDescriptor(&desc));
    if (dropout == 0.0) {
      // this->state_size
      CUDNN(DropoutGetStatesSize(handle.handle, &state_size));
      // this->states
      CUDA(Malloc(&states, state_size));
      CUDNN(SetDropoutDescriptor(
        /*dropoutDesc=*/desc,
        /*handle=*/handle.handle,
        /*dropout=*/dropout,
        /*states=*/states,
        /*stateSizeInBytes=*/state_size,
        /*seed=*/(seed == 0ULL ? time(NULL) : seed))
      );
    } else {
      this->state_size = 0;
      this->states = nullptr;
    }
  }
  ~Dropout() {
    if (desc != nullptr) {
      CUDNN(DestroyDropoutDescriptor(this->desc));
    }
    if (states != nullptr) {
      CUDA(Free(states));
    }
  }
};
