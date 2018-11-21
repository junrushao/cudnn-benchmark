#pragma once

#include <cudnn.h>

#include "macro.h"

namespace cudnn_enums_auto_export {

#define DECLARE_ENUM_INSIDE(type, name, cudnn_name) static const type name
#define DECLARE_ENUM_OUTSIDE(type, name, cudnn_name) const type type::name = type(CUDNN_##cudnn_name);
#define SWITCH_DEFAULT(type) default: throw std::runtime_error(#type ": Not implemented");
#define SWITCH_RETURN(cudnn_name, value) case CUDNN_##cudnn_name : return value;
#define SWITCH_STRING(type, name, cudnn_name) case CUDNN_##cudnn_name : return #type "::" #name;
#define DECLARE_COMMON_FIELDS(type, cudnn_type)           \
  public:                                                 \
    cudnn##cudnn_type##_t v;                              \
    bool operator == (const type &rhs) const {            \
      return v == rhs.v;                                  \
    }                                                     \
  private:                                                \
    explicit type(cudnn##cudnn_type##_t v): v(v) {}


struct DType {
public:
  DECLARE_COMMON_FIELDS(DType, DataType);
public:
  n_bytes_t size_of() const {
    switch (this->v) {
      SWITCH_RETURN(DATA_FLOAT, 4);
      SWITCH_RETURN(DATA_DOUBLE, 8);
      SWITCH_RETURN(DATA_HALF, 2);
      SWITCH_RETURN(DATA_INT8, 1);
      SWITCH_RETURN(DATA_INT32, 4);
      SWITCH_RETURN(DATA_INT8x4, 4);
      SWITCH_RETURN(DATA_UINT8, 1);
      SWITCH_RETURN(DATA_UINT8x4, 4);
      SWITCH_RETURN(DATA_INT8x32, 32);
      SWITCH_DEFAULT();
    }
  }
  std::string str() const {
    switch (this->v) {
      SWITCH_STRING(DType, kFloat, DATA_FLOAT);
      SWITCH_STRING(DType, kDouble, DATA_DOUBLE);
      SWITCH_STRING(DType, kHalf, DATA_HALF);
      SWITCH_STRING(DType, kInt8, DATA_INT8);
      SWITCH_STRING(DType, kInt32, DATA_INT32);
      SWITCH_STRING(DType, kInt8x4, DATA_INT8x4);
      SWITCH_STRING(DType, kUint8, DATA_UINT8);
      SWITCH_STRING(DType, kUint8x4, DATA_UINT8x4);
      SWITCH_STRING(DType, kInt8x32, DATA_INT8x32);
      SWITCH_DEFAULT();
    }
  }
public:
  DECLARE_ENUM_INSIDE(DType, kFloat, DATA_FLOAT);
  DECLARE_ENUM_INSIDE(DType, kDouble, DATA_DOUBLE);
  DECLARE_ENUM_INSIDE(DType, kHalf, DATA_HALF);
  DECLARE_ENUM_INSIDE(DType, kInt8, DATA_INT8);
  DECLARE_ENUM_INSIDE(DType, kInt32, DATA_INT32);
  DECLARE_ENUM_INSIDE(DType, kInt8x4, DATA_INT8x4);
  DECLARE_ENUM_INSIDE(DType, kUint8, DATA_UINT8);
  DECLARE_ENUM_INSIDE(DType, kUint8x4, DATA_UINT8x4);
  DECLARE_ENUM_INSIDE(DType, kInt8x32, DATA_INT8x32);
};
DECLARE_ENUM_OUTSIDE(DType, kFloat, DATA_FLOAT);
DECLARE_ENUM_OUTSIDE(DType, kDouble, DATA_DOUBLE);
DECLARE_ENUM_OUTSIDE(DType, kHalf, DATA_HALF);
DECLARE_ENUM_OUTSIDE(DType, kInt8, DATA_INT8);
DECLARE_ENUM_OUTSIDE(DType, kInt32, DATA_INT32);
DECLARE_ENUM_OUTSIDE(DType, kInt8x4, DATA_INT8x4);
DECLARE_ENUM_OUTSIDE(DType, kUint8, DATA_UINT8);
DECLARE_ENUM_OUTSIDE(DType, kUint8x4, DATA_UINT8x4);
DECLARE_ENUM_OUTSIDE(DType, kInt8x32, DATA_INT8x32);


struct TensorFormat {
public:
  DECLARE_COMMON_FIELDS(TensorFormat, TensorFormat);
public:
  std::string str() const {
    switch (this->v) {
      SWITCH_STRING(TensorFormat, kNCHW, TENSOR_NCHW);
      SWITCH_STRING(TensorFormat, kNHWC, TENSOR_NHWC);
      SWITCH_STRING(TensorFormat, kNCHWVectC, TENSOR_NCHW_VECT_C);
      SWITCH_DEFAULT();
    }
  }
public:
  DECLARE_ENUM_INSIDE(TensorFormat, kNCHW, TENSOR_NCHW);
  DECLARE_ENUM_INSIDE(TensorFormat, kNHWC, TENSOR_NHWC);
  DECLARE_ENUM_INSIDE(TensorFormat, kNCHWVectC, TENSOR_NCHW_VECT_C);
  DECLARE_ENUM_INSIDE(TensorFormat, kUndefined, UNDEFINED);
};
DECLARE_ENUM_OUTSIDE(TensorFormat, kNCHW, TENSOR_NCHW);
DECLARE_ENUM_OUTSIDE(TensorFormat, kNHWC, TENSOR_NHWC);
DECLARE_ENUM_OUTSIDE(TensorFormat, kNCHWVectC, TENSOR_NCHW_VECT_C);


namespace RNN {
struct Algo {
public:
  DECLARE_COMMON_FIELDS(Algo, RNNAlgo);
public:
  std::string str() const {
    switch (this->v) {
      SWITCH_STRING(RNNAlgo, kStandard, RNN_ALGO_STANDARD);
      SWITCH_STRING(RNNAlgo, kPersistStatic, RNN_ALGO_PERSIST_STATIC);
      SWITCH_STRING(RNNAlgo, kPersistDynamic, RNN_ALGO_PERSIST_DYNAMIC);
      SWITCH_DEFAULT();
    }
  }
public:
  DECLARE_ENUM_INSIDE(Algo, kStandard, RNN_ALGO_STANDARD);
  DECLARE_ENUM_INSIDE(Algo, kPersistStatic, RNN_ALGO_PERSIST_STATIC);
  DECLARE_ENUM_INSIDE(Algo, kPersistDynamic, RNN_ALGO_PERSIST_DYNAMIC);
};
DECLARE_ENUM_OUTSIDE(Algo, kStandard, RNN_ALGO_STANDARD);
DECLARE_ENUM_OUTSIDE(Algo, kPersistStatic, RNN_ALGO_PERSIST_STATIC);
DECLARE_ENUM_OUTSIDE(Algo, kPersistDynamic, RNN_ALGO_PERSIST_DYNAMIC);


struct InputMode {
public:
  DECLARE_COMMON_FIELDS(InputMode, RNNInputMode);
public:
  std::string str() const {
    switch (this->v) {
      SWITCH_STRING(InputMode, kLinearInput, LINEAR_INPUT);
      SWITCH_STRING(InputMode, kSkipInput, SKIP_INPUT);
      SWITCH_DEFAULT();
    }
  }
public:
  DECLARE_ENUM_INSIDE(InputMode, kLinearInput, LINEAR_INPUT);
  DECLARE_ENUM_INSIDE(InputMode, kSkipInput, SKIP_INPUT);
};
DECLARE_ENUM_OUTSIDE(InputMode, kLinearInput, LINEAR_INPUT);
DECLARE_ENUM_OUTSIDE(InputMode, kSkipInput, SKIP_INPUT);


struct Direction {
public:
  DECLARE_COMMON_FIELDS(Direction, DirectionMode);
public:
  std::string str() const {
    switch (this->v) {
      SWITCH_STRING(Direction, kUnidirectional, UNIDIRECTIONAL);
      SWITCH_STRING(Direction, kBidirectional, BIDIRECTIONAL);
      SWITCH_DEFAULT();
    }
  }
  n_bytes_t n_dir() const {
    switch (this->v) {
      SWITCH_RETURN(UNIDIRECTIONAL, 1);
      SWITCH_RETURN(BIDIRECTIONAL, 2);
      SWITCH_DEFAULT();
    }
  }
public:
  DECLARE_ENUM_INSIDE(Direction, kUnidirectional, UNIDIRECTIONAL);
  DECLARE_ENUM_INSIDE(Direction, kBidirectional, BIDIRECTIONAL);
};
DECLARE_ENUM_OUTSIDE(Direction, kUnidirectional, UNIDIRECTIONAL);
DECLARE_ENUM_OUTSIDE(Direction, kBidirectional, BIDIRECTIONAL);


struct CellType {
public:
  DECLARE_COMMON_FIELDS(CellType, RNNMode);
public:
  std::string str() const {
    switch (this->v) {
      SWITCH_STRING(CellType, kReLU, RNN_RELU);
      SWITCH_STRING(CellType, kTanH, RNN_TANH);
      SWITCH_STRING(CellType, kGRU, GRU);
      SWITCH_STRING(CellType, kLSTM, LSTM);
      SWITCH_DEFAULT();
    }
  }
public:
  DECLARE_ENUM_INSIDE(CellType, kReLU, RNN_RELU);
  DECLARE_ENUM_INSIDE(CellType, kTanH, RNN_TANH);
  DECLARE_ENUM_INSIDE(CellType, kGRU, GRU);
  DECLARE_ENUM_INSIDE(CellType, kLSTM, LSTM);
};
DECLARE_ENUM_OUTSIDE(CellType, kReLU, RNN_RELU);
DECLARE_ENUM_OUTSIDE(CellType, kTanH, RNN_TANH);
DECLARE_ENUM_OUTSIDE(CellType, kGRU, GRU);
DECLARE_ENUM_OUTSIDE(CellType, kLSTM, LSTM);

}  // namespace RNN

#undef DECLARE_COMMON_FIELDS
#undef SWITCH_STRING
#undef SWITCH_RETURN
#undef SWITCH_DEFAULT
#undef DECLARE_ENUM_OUTSIDE
#undef DECLARE_ENUM_INSIDE

}  // namespace cudnn_enums_auto_export
