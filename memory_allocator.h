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
  template <typename T>
  void visit_state(const Context &ctx, T &v);
public:
  void visit_state(const Context &ctx, TensorU &v) {
    alloc(v->data, v->size * v->dtype.size_of());
  }
  void visit_state(const Context &ctx, FilterU &v) {
    alloc(v->data, v->size * v->dtype.size_of());
  }
  void visit_state(const Context &ctx, DropoutU &v) {
    n_bytes_t state_size = v->get_state_size(ctx);
    alloc(v->state, state_size);
    CUDNN(SetDropoutDescriptor(
      /*dropoutDesc=*/v->desc.get(),
      /*handle=*/ctx.get(),
      /*dropout=*/v->config.dropout,
      /*states=*/v->state,
      /*stateSizeInBytes=*/state_size,
      /*seed=*/(v->config.seed == 0ULL ? time(NULL) : v->config.seed))
    );
  }
  void visit_state(const Context &ctx, RNNU &v) {
    visit_state(ctx, v->filter);
  }
  void visit_state(const Context &ctx, RNNStruct::State &v) {
    if (v.h != nullptr) {
      visit_state(ctx, v.h);
    }
    if (v.c != nullptr) {
      visit_state(ctx, v.c);
    }
  }
  void visit_state(const Context &ctx, SeqU &v) {
    alloc(v->data, v->size * v->dtype.size_of());
  }
  void visit_workspace(const Context &ctx, const SeqU &seq, RNNU &v) {
    size_t _workspace_size = 0;
    CUDNN(GetRNNWorkspaceSize(
      /*handle=*/ctx.get(),
      /*cudnnRNNDescriptor_t=*/v->desc.get(),
      /*seqLength=*/seq->raw_steps.size(),
      /*xDesc=*/seq->raw_steps.data(),
      /*sizeInBytes=*/&_workspace_size
    ));
    n_bytes_t workspace_size = static_cast<n_bytes_t>(_workspace_size);
    v->workspace_size = workspace_size;
    alloc(v->workspace, workspace_size);
  }
};
