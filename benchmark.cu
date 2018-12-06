#include <iostream>
#include "timer.h"
#include "util.h"
#include "assert.h"

using DType = cudnn_enums_auto_export::DType;
using Context = cudnn_handles_auto_export::Context;

struct RunConfig {
  int seq_len{1};
  int hidden_size{2};
  RNNStruct::CellType cell{RNNStruct::CellType::kLSTM};
  RNNStruct::Algo algo{RNNStruct::Algo::kStandard};
  int n_layers{1};
  int batch_size{1};
  RNNStruct::InputMode input_mode{RNNStruct::InputMode::kLinearInput};
  DType dtype{DType::kFloat};
};

double eval(RunConfig rconfig) {
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
  // warm up
  constexpr int WARMUP_TIMES = 5;
  for (int i = 0; i < WARMUP_TIMES; ++i) {
    rnn->forward_inference(ctx, seqX, stateX, seqY, stateY);
  }
  // evaluate
  constexpr int EVAL_TIMES = 100;
  const double start_time = CycleTimer::currentSeconds();
  for (int i = 0; i < EVAL_TIMES; ++i) {
    rnn->forward_inference(ctx, seqX, stateX, seqY, stateY);
  }
  const double end_time = CycleTimer::currentSeconds();
  return (end_time - start_time) / EVAL_TIMES;
}

void run() {
  using CellType = RNNStruct::CellType;
  using Algo = RNNStruct::Algo;
  int counter = 0;
  std::cerr << "idx,seq_len,hidden_size,cell,algo,time(ms)" << std::endl;
  for (int seq_len : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512})
  for (int hidden_size: {1, 2, 4, 8, 16, 32, 64, 128, 256, 512})
  for (CellType cell: {CellType::kTanH, CellType::kGRU, CellType::kLSTM})
  for (Algo algo: {Algo::kStandard, Algo::kPersistStatic}) {
    if (true) {
      RunConfig rconfig;
      rconfig.seq_len = seq_len;
      rconfig.hidden_size = hidden_size;
      rconfig.cell = cell;
      rconfig.algo = algo;
      double eval_time = eval(rconfig);
      std::cout << counter << ','
                << seq_len << ','
                << hidden_size << ','
                << cell.str() << ','
                << algo.str() << ','
                << eval_time * 1000.0
                << std::endl;
    }
    counter += 1;
    if (counter % 100 == 0) {
      std::cerr << "Have finished " << counter << " tasks" << std::endl;
    }
  }
}

int main() {
  srand(1241323);
  run();
  return 0;
}
