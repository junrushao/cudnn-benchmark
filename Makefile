CXX := nvcc
TARGET := main
CUDA_PATH := $(CUDA_HOME)
CUDNN_PATH := $(CUDNN_HOME)
HEADERS := -I $(CUDNN_PATH)/include
LIBS := -L $(CUDNN_PATH)/lib64 -L $(CUDA_PATH)/lib64
CXXFLAGS := -arch=sm_61 -std=c++11 -O3 --compiler-options -Wall -lcudnn -lcuda

all: main

main: $(TARGET).cu rnn.h util.h
	$(CXX) $(CXXFLAGS) $(HEADERS) $(LIBS) $(TARGET).cu -o $(TARGET)

.phony: clean

clean:
	rm $(TARGET) || echo -n ""
