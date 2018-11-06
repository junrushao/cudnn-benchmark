#include <cassert>
#include <cstdlib>
#include <iostream>
#include <unordered_map>
#include "timer.h"
#include "rnn.h"


template <typename T>
__global__ void _fill_kernel(T *array, int size, T value) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int step = gridDim.x * blockDim.x;
  for (int i = idx; i < size; i += step) {
    array[i] = value;
  }
}


template <typename T>
void fill(void *array, int size, T value) {
  T *_array = static_cast<T*>(array);
  const int blockSize = 256;
  const int numBlocks = (size + blockSize - 1) / blockSize;
  _fill_kernel<<<numBlocks, blockSize>>>(_array, size, value);
}


void run(int seq_len, const Rnn::Config &config) {
  DnnHandle handle;
  Rnn rnn(handle, config);
  rnn.alloc();
  TensorDesc *hcDescs = new TensorDesc[4];
  for (int i = 0; i < 4; ++i) {
    rnn.getHCDesc(hcDescs + i);
    hcDescs[i].alloc();
  }
  TensorDesc &hxDesc = hcDescs[0];
  TensorDesc &hyDesc = hcDescs[1];
  TensorDesc &cxDesc = hcDescs[2];
  TensorDesc &cyDesc = hcDescs[3];
  TensorDesc *xDesc = new TensorDesc[seq_len];
  TensorDesc *yDesc = new TensorDesc[seq_len];
  for (int i = 0; i < seq_len; ++i) {
    rnn.getXDesc(xDesc + i);
    rnn.getYDesc(yDesc + i);
  }
  void *x;
  void *y;
  std::cout << "xDesc[0].data_bytes = " << xDesc[0].data_bytes << std::endl;
  std::cout << "yDesc[0].data_bytes = " << yDesc[0].data_bytes << std::endl;
  CUDA(Malloc(&x, xDesc[0].data_bytes * seq_len));
  CUDA(Malloc(&y, yDesc[0].data_bytes * seq_len));

  const int cold_start_times = 1000;
  const int benchmark_times = 10000;
  for (int i = 0; i < cold_start_times; ++i) {
    rnn.forward_inference(
      /*handle=*/handle,
      /*seq_len=*/seq_len,
      /*xDesc=*/xDesc,
      /*x=*/x,
      /*hxDesc=*/hxDesc,
      /*cxDesc=*/cxDesc,
      /*yDesc=*/yDesc,
      /*y=*/y,
      /*hyDesc=*/hyDesc,
      /*cyDesc=*/cyDesc
    );
  }
  std::cout << "Start benchmarking!" << std::endl;
  double startTime = CycleTimer::currentSeconds();
  for (int i = 0; i < benchmark_times; ++i) {
    rnn.forward_inference(
      /*handle=*/handle,
      /*seq_len=*/seq_len,
      /*xDesc=*/xDesc,
      /*x=*/x,
      /*hxDesc=*/hxDesc,
      /*cxDesc=*/cxDesc,
      /*yDesc=*/yDesc,
      /*y=*/y,
      /*hyDesc=*/hyDesc,
      /*cyDesc=*/cyDesc
    );
  }
  double endTime = CycleTimer::currentSeconds();
  printf("Average time used: %.4f ms\n", (endTime - startTime) * 1000 / benchmark_times);
}


int main(int argc, const char* argv[]) {
  CUDA(SetDevice(/*gpu_id=*/0));
  Rnn::Config config;
  int seq_len;
  if (argc == 8) { // parse_args
    seq_len = std::atoi(argv[1]);
    config.batch_size = std::atoi(argv[2]);
    config.input_size = config.hidden_size = std::atoi(argv[3]);
    config.n_layers = std::atoi(argv[4]);
    config.input = Rnn::Input::kSkipInput;
    config.direction = std::unordered_map<std::string, Rnn::Direction>{
      {"1", Rnn::Direction::kUnidirectional},
      {"2", Rnn::Direction::kBidirectional}}[argv[5]];
    config.mode = std::unordered_map<std::string, Rnn::Mode>{
      {"tanh", Rnn::Mode::kRnnTanh},
      {"relu", Rnn::Mode::kRnnRelu},
      {"lstm", Rnn::Mode::kLstm},
      {"gru", Rnn::Mode::kGru}}[argv[6]];
    config.algo = std::unordered_map<std::string, Rnn::Algo>{
      {"std", Rnn::Algo::kStandard},
      {"ps", Rnn::Algo::kPersistStatic},
      {"pd", Rnn::Algo::kPersistDynamic}}[argv[7]];
    config.dtype = DType::kFloat;
  } else {
    puts("Usage: ./main [seq_len] [batch_size] [hidden_size] [n_layers] [direction] [mode] [algo]");
    return 1;
  }
  std::cout << "==============================" << std::endl;
  std::cout << "Sequence length = " << seq_len << std::endl;
  std::cout << "Batch size = " << config.batch_size << std::endl;
  std::cout << "Hidden size = " << config.hidden_size << std::endl;
  std::cout << "Number of layers = " << config.n_layers << std::endl;
  std::cout << "Input mode = " << to_string(config.input) << std::endl;
  std::cout << "Direction = " << to_string(config.direction) << std::endl;
  std::cout << "RNN mode = " << to_string(config.mode) << std::endl;
  std::cout << "Algo = " << to_string(config.algo) << std::endl;
  std::cout << "==============================" << std::endl;
  run(seq_len, config);
  return 0;
}
