CXX := nvcc
TARGET := main
TARGET2 := benchmark
CUDA_PATH := $(CUDA_HOME)
CUDNN_PATH := $(CUDNN_HOME)
HEADERS := -I $(CUDNN_PATH)/include
LIBS := -L $(CUDNN_PATH)/lib64 -L $(CUDA_PATH)/lib64
CXXFLAGS := -arch=sm_70 -std=c++14 -O3 --compiler-options -Wall --compiler-options -Wextra -lcudnn -lcuda
# CXXFLAGS := -arch=sm_70 -std=c++14 -O0 -g --compiler-options -g3 --compiler-options -Wall --compiler-options -Wextra -lcudnn -lcuda

all: main benchmark

main: $(TARGET).cu *.h
	$(CXX) $(CXXFLAGS) $(HEADERS) $(LIBS) $(TARGET).cu -o $(TARGET)

benchmark: $(TARGET2).cu *.h
	$(CXX) $(CXXFLAGS) $(HEADERS) $(LIBS) $(TARGET2).cu -o $(TARGET2)

.phony: clean

clean:
	rm $(TARGET) || echo -n ""
	rm $(TARGET2) || echo -n ""
