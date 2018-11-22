#include <iostream>
#include "timer.h"
#include "util.h"

using DType = cudnn_enums_auto_export::DType;
using Context = cudnn_handles_auto_export::Context;

struct RunConfig {
  int seq_len{0};
  int hidden_size{0};
  RNNStruct::CellType cell{RNNStruct::CellType::kLSTM};
  RNNStruct::Algo algo{RNNStruct::Algo::kStandard};

  int n_layers{1};
  int batch_size{1};
  RNNStruct::InputMode input_mode{RNNStruct::InputMode::kSkipInput};
  DType dtype{DType::kFloat};
};

void run() {
  RunConfig rconfig;
  rconfig.seq_len = 1;
  rconfig.hidden_size = 2;
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
  rnn->forward_inference(ctx, seqX, stateX, seqY, stateY);
}

int main() {
  run();
  return 0;
}
