#pragma once

#include <cudnn.h>

#include "macro.h"
#include "cudnn_enums.h"
#include "cudnn_handles.h"
#include "dropout.h"
#include "tensor.h"

struct RNNStruct {
public:
	using Context = cudnn_handles_auto_export::Context;
	using DType = cudnn_enums_auto_export::DType;
	using TensorFormat = cudnn_enums_auto_export::TensorFormat;
	using InputMode = cudnn_enums_auto_export::RNN::InputMode;
	using Direction = cudnn_enums_auto_export::RNN::Direction;
	using CellType = cudnn_enums_auto_export::RNN::CellType;
	using Algo = cudnn_enums_auto_export::RNN::Algo;
	using RNNDesc = cudnn_handles_auto_export::RNNDesc;
public:
  struct Config {
    int input_size{0};
    int hidden_size{0};
    int n_layers{0};
		DropoutStruct::Config dropout;
    InputMode input_mode{InputMode::kSkipInput};
    Direction direction{Direction::kUnidirectional};
    CellType cell{CellType::kLSTM};
    Algo algo{Algo::kStandard};
  };
	struct State {
		TensorU h{nullptr};
		TensorU c{nullptr};
		State(int n_states, DType dtype, const std::vector<int> &shape) {
			if (n_states == 1) {
				h.reset(new TensorStruct(dtype, shape));
			} else if (n_states == 2) {
				h.reset(new TensorStruct(dtype, shape));
				c.reset(new TensorStruct(dtype, shape));
			} else {
				throw std::runtime_error("Internal error!");
			}
		}
	};
public:
	Config config;
  DropoutU dropout{nullptr};
	RNNDesc desc{nullptr};
	DType dtype{DType::kFloat};
  FilterU filter{nullptr};
	// TODO(@junrushao1994): add cudnnPersistentRNNPlan_t for dynamic persistent mode
  explicit RNNStruct(const Context &ctx,
							 			 const Config &config,
							 			 const DType &dtype):
		config(config),
		dropout(new DropoutStruct(config.dropout)),
		desc(cudnn_handles_auto_export::CreateRNNDesc()),
		dtype(dtype)
	{
    CUDNN(SetRNNDescriptor(
      /*handle=*/ctx.get(),
      /*rnnDesc=*/desc.get(),
      /*hiddenSize=*/config.hidden_size,
      /*numLayers=*/config.n_layers,
      /*dropoutDesc=*/dropout->desc.get(),
      /*inputMode=*/config.input_mode.v,
      /*direction=*/config.direction.v,
      /*mode=*/config.cell.v,
      /*algo=*/config.algo.v,
      /*dataType=*/dtype.v
    ));
		TensorStruct input(dtype, {1, config.input_size, 1});
    size_t filter_bytes = 0;
    CUDNN(GetRNNParamsSize(
      /*handle=*/ctx.get(),
      /*rnnDesc=*/desc.get(),
      /*xDesc=*/input.desc.get(),
      /*sizeInBytes=*/&filter_bytes,
      /*dataType*/dtype.v
    ));
    if (filter_bytes % dtype.size_of() != 0) {
      std::runtime_error("Indivisible!");
    }
		int filter_size = static_cast<int>(filter_bytes);
		filter_size /= dtype.size_of();
		filter.reset(new FilterStruct(dtype, TensorFormat::kNCHW, {filter_size, 1, 1}));
	}
	State get_state(int batch_size) {
		int n_layers = config.n_layers;
		int hidden_size = config.hidden_size;
		int n_dirs = static_cast<int>(config.direction.n_dir());
		if (config.cell == CellType::kLSTM) {
			return State(2, dtype, {n_layers * n_dirs, batch_size, hidden_size});
		} else {
			return State(1, dtype, {n_layers * n_dirs, batch_size, hidden_size});
		}
	}
};
using RNNU = std::unique_ptr<RNNStruct>;
using RNNS = std::shared_ptr<RNNStruct>;