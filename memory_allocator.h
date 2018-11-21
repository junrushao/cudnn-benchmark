#pragma once

#include <time.h>

#include "cudnn_handles.h"
#include "cudnn_enums.h"

#include "./macro.h"
#include "./tensor.h"
#include "./filter.h"
#include "./dropout.h"
#include "./rnn.h"
#include "./seq.h"


struct MemoryAllocator {
public:
  using Context = cudnn_handles_auto_export::Context;
public:
  struct CUDAMemoryDeleter {
    void operator () (void *ptr) const {
      CUDA(Free(ptr));
    };
  };
  using cudaPtr = std::unique_ptr<void, CUDAMemoryDeleter>;
public:
  std::vector<cudaPtr> memory_chunks;
public:
  void alloc(void *&data, n_bytes_t size) {
    if (size > 0) {
      CUDA(Malloc(&data, size));
      memory_chunks.emplace_back(data);
    } else {
      data = nullptr;
    }
  }
public:
  void visit(const Context &ctx, TensorStruct &v) {
    alloc(v.data, v.size * v.dtype.size_of());
  }
  void visit(const Context &ctx, FilterStruct &v) {
    alloc(v.data, v.size * v.dtype.size_of());
  }
  void visit(const Context &ctx, DropoutStruct &v) {
    n_bytes_t state_size = v.get_state_size(ctx);
    alloc(v.state, state_size);
    CUDNN(SetDropoutDescriptor(
        /*dropoutDesc=*/v.desc.get(),
        /*handle=*/ctx.get(),
        /*dropout=*/v.config.dropout,
        /*states=*/v.state,
        /*stateSizeInBytes=*/state_size,
        /*seed=*/(v.config.seed == 0ULL ? time(NULL) : v.config.seed))
      );
  }
  void visit(const Context &ctx, RNNStruct &v) {
    visit(ctx, *v.filter);
  }
  void visit(const Context &ctx, RNNStruct::State &v) {
    if (v.h != nullptr) {
      visit(ctx, *v.h);
    }
    if (v.c != nullptr) {
      visit(ctx, *v.c);
    }
  }
  void visit(const Context &ctx, SeqStruct &v) {
    alloc(v.data, v.size * v.dtype.size_of());
  }
};
