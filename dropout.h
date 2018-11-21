#pragma once

#include <cudnn.h>
#include <memory>
#include <time.h>

#include "cudnn_handles.h"
#include "cudnn_enums.h"

struct DropoutStruct {
public:
  using Context = cudnn_handles_auto_export::Context;
  using DropoutDesc = cudnn_handles_auto_export::DropoutDesc;
public:
  struct Config {
    float dropout{0.0};
    seed_t seed{0};
  };
public:
  DropoutDesc desc{nullptr};
  Config config;
public:
  DropoutStruct(const Config &config):
    desc(cudnn_handles_auto_export::CreateDropoutDesc()),
    config(config)
  {
  }
  ~DropoutStruct() {}
};
using DropoutU = std::unique_ptr<DropoutStruct>;
using DropoutS = std::shared_ptr<DropoutStruct>;
