CXX := nvcc
TARGET := main
CUDA_PATH := $(CUDA_HOME)
CUDNN_PATH := $(CUDNN_HOME)
HEADERS := -I $(CUDNN_PATH)/include
LIBS := -L $(CUDNN_PATH)/lib64 -L $(CUDA_PATH)/lib64
CXXFLAGS := -arch=sm_70 -std=c++14 -O3 --compiler-options -Wall --compiler-options -Wextra -lcudnn -lcuda

all: main

main: $(TARGET).cu *.h
	$(CXX) $(CXXFLAGS) $(HEADERS) $(LIBS) $(TARGET).cu -o $(TARGET)

.phony: clean

clean:
	rm $(TARGET) || echo -n ""
