#include <iostream>
#include "timer.h"
#include "util.h"
#include "assert.h"

using DType = cudnn_enums_auto_export::DType;
using Context = cudnn_handles_auto_export::Context;

struct RunConfig {
  int seq_len{1};
  int hidden_size{2};
  RNNStruct::CellType cell{RNNStruct::CellType::kTanH};
  RNNStruct::Algo algo{RNNStruct::Algo::kStandard};
  int n_layers{1};
  int batch_size{1};
  RNNStruct::InputMode input_mode{RNNStruct::InputMode::kSkipInput};
  DType dtype{DType::kFloat};
};

void run(int seq_len) {
  RunConfig rconfig;
  rconfig.seq_len = seq_len;
  Context ctx = cudnn_handles_auto_export::CreateContext();
  RNNU rnn = std::make_unique<RNNStruct>(
    ctx,
    RNNStruct::Config{
      /*input_size=*/rconfig.hidden_size,
      /*hidden_size=*/rconfig.hidden_size,
      /*n_layers=*/rconfig.n_layers,
      /*dropout=*/DropoutStruct::Config{0.0, 0},
      /*input_mode=*/rconfig.input_mode,
      /*direction=*/RNNStruct::Direction::kUnidirectional,
      /*cell=*/rconfig.cell,
      /*algo=*/rconfig.algo
    },
    DType::kFloat
  );
  SeqU seqX = std::make_unique<SeqStruct>(rconfig.seq_len, rconfig.batch_size, rconfig.hidden_size, rconfig.dtype);
  SeqU seqY = std::make_unique<SeqStruct>(rconfig.seq_len, rconfig.batch_size, rconfig.hidden_size, rconfig.dtype);
  RNNStruct::State stateX(rnn->get_state(rconfig.batch_size));
  RNNStruct::State stateY(rnn->get_state(rconfig.batch_size));
  MemoryAllocator allocator;
  allocator.visit_state(ctx, seqX);
  allocator.visit_state(ctx, seqY);
  allocator.visit_state(ctx, stateX);
  allocator.visit_state(ctx, stateY);
  allocator.visit_state(ctx, rnn);
  allocator.visit_workspace(ctx, seqX, rnn);

  void *data;
  n_bytes_t size;
  std::vector<int> shape;

  std::cout << "# =========================================================" << std::endl;
  std::cout << "w = [None] * " << rconfig.cell.n_region_per_layer() << std::endl;
  std::cout << "b = [None] * " << rconfig.cell.n_region_per_layer() << std::endl;
  for (int i = 0; i < rconfig.cell.n_region_per_layer(); ++i) {
    rnn->get_weight_region(ctx, 0, i, data, size, shape);
    assert(shape.size() == 3 && shape[0] <= 1);
    shape = {shape[1], shape[2]};
    copyToDevice<float>(data, size, gen_rand_vector<float>(size));
    if (size == 0) {
      std::cout << "w[" << i << + "] = None" << std::endl;
    } else {
      print<float>("w[" + std::to_string(i) + "]", shape, data);
    }
  }
  for (int i = 0; i < rconfig.cell.n_region_per_layer(); ++i) {
    rnn->get_bias_region(ctx, 0, i, data, size, shape);
    assert(shape.size() == 3 && shape[0] == 1 && shape[2] == 1);
    shape = {shape[1]};
    copyToDevice<float>(data, size, gen_rand_vector<float>(size));
    print<float>("b[" + std::to_string(i) + "]", shape, data);
  }
  std::cout << "# =========================================================" << std::endl;
  if (stateX.h) {
    data = stateX.h->data;
    // print<float>("hx", {rconfig.n_layers * /*n_dirs=*/1, rconfig.batch_size, rconfig.hidden_size}, data);
    // assume n_layers = 1, n_dirs = 1, batch_size = 1
    print<float>("hx", {rconfig.hidden_size}, data);
  } else {
    std::cout << "hx = None" << std::endl;
  }
  if (stateX.c) {
    data = stateX.c->data;
    // print<float>("cx", {rconfig.n_layers * /*n_dirs=*/1, rconfig.batch_size, rconfig.hidden_size}, data);
    // assume n_layers = 1, n_dirs = 1, batch_size = 1
    print<float>("cx", {rconfig.hidden_size}, data);
  } else {
    std::cout << "cx = None" << std::endl;
  }
  std::cout << "seq_x = [None] * " << rconfig.seq_len << std::endl;
  for (int i = 0; i < rconfig.seq_len; ++i) {
    seqX->get_step_region(i, data, size);
    assert(size == rconfig.batch_size * rconfig.hidden_size);
    copyToDevice<float>(data, size, gen_rand_vector<float>(rconfig.batch_size * rconfig.hidden_size));
    // copyToDevice<float>(data, size, gen_rand_vector<float>(rconfig.batch_size * rconfig.hidden_size));
    // print<float>("seq_x[" + std::to_string(i) + "]", {rconfig.batch_size, rconfig.hidden_size}, data);
    // assume batch_size = 1
    print<float>("seq_x[" + std::to_string(i) + "]", {rconfig.hidden_size}, data);
  }
  std::cout << "# =========================================================" << std::endl;
  rnn->forward_inference(ctx, seqX, stateX, seqY, stateY);
  if (stateY.h) {
    data = stateY.h->data;
    // print<float>("hy", {rconfig.n_layers * /*n_dirs=*/1, rconfig.batch_size, rconfig.hidden_size}, data);
    // assume n_layers = 1, n_dirs = 1, batch_size = 1
    print<float>("hy", {rconfig.hidden_size}, data);
  } else {
    std::cout << "hy = None" << std::endl;
  }
  if (stateY.c) {
    data = stateY.c->data;
    // print<float>("cy", {rconfig.n_layers * /*n_dirs=*/1, rconfig.batch_size, rconfig.hidden_size}, data);
    // assume n_layers = 1, n_dirs = 1, batch_size = 1
    print<float>("cy", {rconfig.hidden_size}, data);
  } else {
    std::cout << "cy = None" << std::endl;
  }
  std::cout << "seq_y = [None] * " << rconfig.seq_len << std::endl;
  for (int i = 0; i < rconfig.seq_len; ++i) {
    seqY->get_step_region(i, data, size);
    assert(size == rconfig.batch_size * rconfig.hidden_size);
    // print<float>("seq_y[" + std::to_string(i) + "]", {rconfig.batch_size, rconfig.hidden_size}, data);
    // assume batch_size = 1
    print<float>("seq_y[" + std::to_string(i) + "]", {rconfig.hidden_size}, data);
  }
}

int main(int argc, char **argv) {
  srand(1241323);
  run(std::atoi(argv[1]));
  return 0;
}
