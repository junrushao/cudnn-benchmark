#pragma once

#include <cudnn.h>
#include <sstream>
#include <memory>
#include <vector>


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


#define AS(type, data) (static_cast<type>((data)))


struct DnnHandle {
  NO_COPY_MOVE_ASSIGN(DnnHandle);
public:
  cudnnHandle_t handle;
  explicit DnnHandle() {
    CUDNN(Create(&handle));
  }
  ~DnnHandle() {
    CUDNN(Destroy(handle));
  }
};


enum class DType {
  kFloat = CUDNN_DATA_FLOAT,      // = 0
  kDouble = CUDNN_DATA_DOUBLE,    // = 1
  kHalf = CUDNN_DATA_HALF,        // = 2
  kInt8 = CUDNN_DATA_INT8,        // = 3
  kInt32 = CUDNN_DATA_INT32,      // = 4
  kInt8x4 = CUDNN_DATA_INT8x4,    // = 5
  kUint8 = CUDNN_DATA_UINT8,      // = 6
  kUint8x4 = CUDNN_DATA_UINT8x4,  // = 7
  kInt8x32 = CUDNN_DATA_INT8x32,  // = 8
};

std::string to_string(const DType &v) {
  switch (v) {
  case DType::kFloat:
    return "DType::kFloat";
  case DType::kDouble:
    return "DType::kDouble";
  case DType::kHalf:
    return "DType::kHalf";
  case DType::kInt8:
    return "DType::kInt8";
  case DType::kInt32:
    return "DType::kInt32";
  case DType::kInt8x4:
    return "DType::kInt8x4";
  case DType::kUint8:
    return "DType::kUint8";
  case DType::kUint8x4:
    return "DType::kUint8x4";
  case DType::kInt8x32:
    return "DType::kInt8x32";
  default:
    throw std::runtime_error("Not implemented");
  }
}


int Sizeof(const cudnnDataType_t &dtype) {
  switch (dtype) {
  case CUDNN_DATA_FLOAT:
    return sizeof(float);
  case CUDNN_DATA_DOUBLE:
    return sizeof(double);
  case CUDNN_DATA_HALF:
    return sizeof(float) / 2;
  case CUDNN_DATA_INT8:
    return sizeof(int8_t);
  case CUDNN_DATA_INT32:
    return sizeof(int32_t);
  case CUDNN_DATA_INT8x4:
    return sizeof(int32_t);
  case CUDNN_DATA_UINT8:
    return sizeof(u_int8_t);
  case CUDNN_DATA_UINT8x4:
    return sizeof(u_int32_t);
  case CUDNN_DATA_INT8x32:
    return sizeof(__int128_t);
  default:
    throw std::runtime_error("Not implemented");
  }
}

int Sizeof(const DType &dtype) {
  return Sizeof(AS(cudnnDataType_t, dtype));
}


enum class TensorFormat {
  kNchw = CUDNN_TENSOR_NCHW,              // = 0
  kNhwc = CUDNN_TENSOR_NHWC,              // = 1
  kNchwVectC = CUDNN_TENSOR_NCHW_VECT_C,  // = 2
};


struct TensorDesc {
  NO_COPY_MOVE_ASSIGN(TensorDesc);
public:
  cudnnTensorDescriptor_t desc = nullptr;
  size_t data_bytes = 0;
  void *data = nullptr;
public:
  explicit TensorDesc() {
  }
  explicit TensorDesc(const DType &dtype,
                      const std::vector<int> &shape) {
    init(dtype, shape);
  }
  void init(const DType &dtype,
            const std::vector<int> &shape) {
    int ndim = shape.size();
    std::vector<int> stride(ndim);
    data_bytes = 1;
    for (int i = ndim - 1; i >= 0; --i) {
      stride[i] = data_bytes;
      data_bytes *= shape[i];
    }
    data_bytes *= Sizeof(dtype);
    CUDNN(CreateTensorDescriptor(&desc));
    CUDNN(SetTensorNdDescriptor(
      /*tensorDesc=*/desc,
      /*dataType=*/AS(cudnnDataType_t, dtype),
      /*nbDims=*/ndim,
      /*dimA=*/shape.data(),
      /*strideA=*/stride.data())
    );
  }
  void deinit() {
    if (desc != nullptr) {
      CUDNN(DestroyTensorDescriptor(desc));
      desc = nullptr;
    }
    data_bytes = 0;
    if (data != nullptr) {
      CUDA(Free(data));
      data = nullptr;
    }
  }
  void alloc() {
    assert(data == nullptr);
    CUDA(Malloc(&data, data_bytes));
  }
  ~TensorDesc() {
    deinit();
  }
};


struct FilterDesc {
  NO_COPY_MOVE_ASSIGN(FilterDesc);
public:
  cudnnFilterDescriptor_t desc = nullptr;
  size_t data_bytes = 0;
  void* data = nullptr;
public:
  explicit FilterDesc() {
  }
  explicit FilterDesc(const DType &dtype,
                      const TensorFormat &format,
                      const std::vector<int> &shape) {
    init(dtype, format, shape);
  }
  void init(const DType &dtype,
            const TensorFormat &format,
            const std::vector<int> &shape) {
    int ndim = shape.size();
    CUDNN(CreateFilterDescriptor(&desc));
    CUDNN(SetFilterNdDescriptor(
      /*filterDesc=*/desc,
      /*dataType=*/AS(cudnnDataType_t, dtype),
      /*format=*/AS(cudnnTensorFormat_t, format),
      /*nbDims=*/ndim,
      /*filterDimA*/shape.data())
    );
    data_bytes = Sizeof(dtype);
    for (int dim : shape) {
      data_bytes *= dim;
    }
  }
  void deinit() {
    if (desc != nullptr) {
      CUDNN(DestroyFilterDescriptor(desc));
      desc = nullptr;
    }
    data_bytes = 0;
    if (data != nullptr) {
      CUDA(Free(data));
      data = nullptr;
    }
  }
  void alloc() {
    assert(data == nullptr);
    CUDA(Malloc(&data, data_bytes));
  }
  ~FilterDesc() {
    deinit();
  }
};
