#pragma once

#include "./util.h"
#include "./dropout.h"

struct Rnn {
  NO_COPY_MOVE_ASSIGN(Rnn);
public:
  enum class Input {
    kLinearInput = CUDNN_LINEAR_INPUT,                  // = 0
    kSkipInput = CUDNN_SKIP_INPUT,                      // = 1
  };
  enum class Direction {
    kUnidirectional = CUDNN_UNIDIRECTIONAL,             // = 0
    kBidirectional = CUDNN_BIDIRECTIONAL,               // = 1
  };
  enum class Mode {
    kRnnRelu = CUDNN_RNN_RELU,                          // = 0
    kRnnTanh = CUDNN_RNN_TANH,                          // = 1
    kLstm = CUDNN_LSTM,                                 // = 2
    kGru = CUDNN_GRU,                                   // = 3
  };
  enum class Algo {
    kStandard = CUDNN_RNN_ALGO_STANDARD,                // = 0
    kPersistStatic = CUDNN_RNN_ALGO_PERSIST_STATIC,     // = 1
    kPersistDynamic = CUDNN_RNN_ALGO_PERSIST_DYNAMIC,   // = 2
  };
public:
  struct ParamRegion {
    void* ptr;
    size_t size;
    ParamRegion(void *ptr, int64_t size): ptr(ptr), size(size) {
    }
  };
  struct Config {
    int batch_size;
    int input_size;
    int hidden_size;
    int n_layers;
    float dropout;
    Input input;
    Direction direction;
    Mode mode;
    Algo algo;
    DType dtype;
  };
public:
  Config config;
  int n_dir;
  Dropout dropout;
  FilterDesc filter_desc;
  cudnnRNNDescriptor_t desc = nullptr;
  cudnnPersistentRNNPlan_t plan = nullptr;

  size_t workspace_size = 0;
  void *workspace = nullptr;
public:
  explicit Rnn(const DnnHandle &handle, const Config &config):
    // this->config
    config(config),
    // this->n_dir
    n_dir(config.direction == Direction::kBidirectional ? 2 : 1),
    // this->dropout
    dropout(handle, config.dropout),
    // uninitialized: this->filter_desc
    filter_desc()
  {
    // this->desc
    CUDNN(CreateRNNDescriptor(&desc));
    CUDNN(SetRNNDescriptor(
      /*handle=*/handle.handle,
      /*rnnDesc=*/desc,
      /*hiddenSize=*/config.hidden_size,
      /*numLayers=*/config.n_layers,
      /*dropoutDesc=*/this->dropout.desc,
      /*inputMode=*/AS(cudnnRNNInputMode_t, config.input),
      /*direction=*/AS(cudnnDirectionMode_t, config.direction),
      /*mode=*/AS(cudnnRNNMode_t, config.mode),
      /*algo=*/AS(cudnnRNNAlgo_t, config.algo),
      /*dataType=*/AS(cudnnDataType_t, config.dtype)
    ));
    // this->plan
    if (config.algo == Algo::kPersistDynamic) {
      CUDNN(CreatePersistentRNNPlan(
        /*rnnDesc=*/desc,
        /*minibatch=*/config.batch_size,
        /*dataType=*/AS(cudnnDataType_t, config.dtype),
        /*plan=*/&plan
      ));
      CUDNN(SetPersistentRNNPlan(
        /*rnnDesc=*/desc,
        /*plan=*/plan
      ));
    }
    // this->filter_desc
    size_t data_bytes = 0;
    TensorDesc input_desc;
    getXDesc(&input_desc);
    CUDNN(GetRNNParamsSize(
      /*handle=*/handle.handle,
      /*rnnDesc=*/desc,
      /*xDesc=*/input_desc.desc,
      /*sizeInBytes=*/&data_bytes,
      /*dataType*/AS(cudnnDataType_t, config.dtype)
    ));
    if (data_bytes % Sizeof(config.dtype) != 0) {
      std::runtime_error("Indivisible!");
    }
    int data_size = static_cast<int>(data_bytes) / Sizeof(config.dtype);
    filter_desc.init(config.dtype, TensorFormat::kNchw, {data_size, 1, 1});
  }
  void alloc() {
    filter_desc.alloc();
  }
  void forward_inference(const DnnHandle &handle,
                         int seq_len,
                         const TensorDesc *xDesc,
                         const void *x,
                         const TensorDesc &hxDesc,
                         const TensorDesc &cxDesc,
                         TensorDesc *yDesc,
                         void *y,
                         TensorDesc &hyDesc,
                         TensorDesc &cyDesc) {
    std::vector<cudnnTensorDescriptor_t> xDescHandles(seq_len);
    std::vector<cudnnTensorDescriptor_t> yDescHandles(seq_len);
    for (int i = 0; i < seq_len; ++i) {
      xDescHandles[i] = xDesc[i].desc;
      yDescHandles[i] = yDesc[i].desc;
    }
    size_t new_workspace_size = 0;
    CUDNN(GetRNNWorkspaceSize(
      /*handle=*/handle.handle,
      /*cudnnRNNDescriptor_t=*/desc,
      /*seqLength=*/seq_len,
      /*xDesc=*/xDescHandles.data(),
      /*sizeInBytes=*/&new_workspace_size
    ));
    if (new_workspace_size > workspace_size) {
      workspace_size = new_workspace_size;
      if (workspace != nullptr) {
        CUDA(Free(workspace));
        workspace = nullptr;
      }
      CUDA(Malloc(&workspace, workspace_size));
    }
    CUDNN(RNNForwardInference(
      /*handle*/handle.handle,
      /*rnnDesc*/desc,
      /*seqLength*/seq_len,
      /*xDesc*/xDescHandles.data(),
      /*x*/x,
      /*hxDesc*/hxDesc.desc,
      /*hx*/hxDesc.data,
      /*cxDesc*/cxDesc.desc,
      /*cx*/cxDesc.data,
      /*wDesc*/filter_desc.desc,
      /*w*/filter_desc.data,
      /*yDesc*/yDescHandles.data(),
      /*y*/y,
      /*hyDesc*/hyDesc.desc,
      /*hy*/hyDesc.data,
      /*cyDesc*/cyDesc.desc,
      /*cy*/cyDesc.data,
      /*workspace*/workspace,
      /*workSpaceSizeInBytes*/workspace_size
    ));
  }
  void calc_offset(const DnnHandle &handle,
                   std::vector<ParamRegion> &weights,
                   std::vector<ParamRegion> &biases) const {
    weights.clear();
    biases.clear();
    // calculate #layers and #regions_per_layer
    const int n_region_per_layer = [&] {
      switch (config.mode) {
      case Mode::kRnnRelu:
        return 2;
      case Mode::kRnnTanh:
        return 2;
      case Mode::kLstm:
        return 8;
      case Mode::kGru:
        return 6;
      default:
        throw std::runtime_error("Not implemented");
      }
    }();
    const int n_layers = config.n_layers * n_dir;
    // calculate the regions for each weight and each bias
    weights.reserve(n_layers * n_region_per_layer);
    biases.reserve(n_layers * n_region_per_layer);
    TensorDesc input_desc;
    getXDesc(&input_desc);
    for (int layer = 0; layer < n_layers; ++layer) {
      for (int region = 0; region < n_region_per_layer; ++region) {
        for (int type = 0; type < 2; ++type) {
          FilterDesc _region_desc;
          cudnnCreateFilterDescriptor(&_region_desc.desc);
          void *_ptr = nullptr;
          cudnnDataType_t _dtype;
          cudnnTensorFormat_t _format;
          int _ndims = 3;
          int _shape[] = {1, 1, 1};
          if (type == 0) {
            CUDNN(GetRNNLinLayerMatrixParams(
              /*handle=*/handle.handle,
              /*rnnDesc=*/desc,
              /*pseudoLayer=*/layer,
              /*xDesc=*/input_desc.desc,
              /*wDesc=*/filter_desc.desc,
              /*w=*/filter_desc.data,
              /*linLayerID=*/region,
              /*linLayerMatDesc=*/_region_desc.desc,
              /*linLayerMat=*/&_ptr
            ));
          } else {
            CUDNN(GetRNNLinLayerBiasParams(
              /*handle=*/handle.handle,
              /*rnnDesc=*/desc,
              /*pseudoLayer=*/layer,
              /*xDesc=*/input_desc.desc,
              /*wDesc=*/filter_desc.desc,
              /*w=*/filter_desc.data,
              /*linLayerID=*/region,
              /*linLayerMatDesc=*/_region_desc.desc,
              /*linLayerMat=*/&_ptr
            ));
          }
          CUDNN(GetFilterNdDescriptor(
            /*wDesc=*/_region_desc.desc,
            /*nbDimsRequested=*/3,
            /*dataType=*/&_dtype,
            /*format=*/&_format,
            /*nbDims=*/&_ndims,
            /*filterDimA=*/_shape
          ));
          (type == 0 ? weights : biases).emplace_back(_ptr, Sizeof(_dtype) * _shape[0] * _shape[1] * _shape[2]);
        }
      }
    }
  }
  ~Rnn() {
    if (plan != nullptr) {
      CUDNN(DestroyPersistentRNNPlan(plan));
    }
    if (desc != nullptr) {
      CUDNN(DestroyRNNDescriptor(desc));
    }
    if (workspace != nullptr) {
      CUDA(Free(workspace));
      workspace = nullptr;
    }
    workspace_size = 0;
  }
  void getXDesc(TensorDesc *desc) const {
    // for x, dx
    desc->deinit();
    desc->init(config.dtype, {config.batch_size, config.input_size, 1});
  }
  void getYDesc(TensorDesc *desc) const {
    // for y, dyd
    desc->deinit();
    desc->init(config.dtype, {config.batch_size, config.hidden_size * n_dir, 1});
  }
  void getHCDesc(TensorDesc *desc) const {
    // for hx, hy, cx, cy, dhx, dhy, dcx, dcy
    desc->deinit();
    desc->init(config.dtype, {config.n_layers * n_dir, config.batch_size, config.hidden_size});
  }
};


std::string to_string(const Rnn::Input &v) {
  switch (v) {
  case Rnn::Input::kLinearInput:
    return "Rnn::Input::kLinearInput";
  case Rnn::Input::kSkipInput:
    return "Rnn::Input::kSkipInput";
  default:
    throw std::runtime_error("Not implemented");
  }
}


std::string to_string(const Rnn::Direction &v) {
  switch (v) {
  case Rnn::Direction::kUnidirectional:
    return "Rnn::Direction::kUnidirectional";
  case Rnn::Direction::kBidirectional:
    return "Rnn::Direction::kBidirectional";
  default:
    throw std::runtime_error("Not implemented");
  }
}

std::string to_string(const Rnn::Mode &v) {
  switch (v) {
  case Rnn::Mode::kRnnRelu:
    return "Rnn::Mode::kRnnRelu";
  case Rnn::Mode::kRnnTanh:
    return "Rnn::Mode::kBidirectional";
  case Rnn::Mode::kLstm:
    return "Rnn::Mode::kLstm";
  case Rnn::Mode::kGru:
    return "Rnn::Mode::kGru";
  default:
    throw std::runtime_error("Not implemented");
  }
}

std::string to_string(const Rnn::Algo &v) {
  switch (v) {
  case Rnn::Algo::kStandard:
    return "Rnn::Algo::kStandard";
  case Rnn::Algo::kPersistStatic:
    return "Rnn::Algo::kPersistStatic";
  case Rnn::Algo::kPersistDynamic:
    return "Rnn::Algo::kPersistDynamic";
  default:
    throw std::runtime_error("Not implemented");
  }
}
