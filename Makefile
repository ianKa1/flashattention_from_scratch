NVCC        := nvcc
CXXFLAGS    := -O3 -std=c++17
CUDA_FLAGS  := -O3 --use_fast_math

CUDA_HOME   := /usr/local/cuda
INCLUDES    := -I$(CUDA_HOME)/include
LIBS        := -L$(CUDA_HOME)/lib64 -lcudart

TARGET      := flash_attn_test

SRCS        := compile_attn_cuda.cpp flash_attn_cuda.cu
OBJS        := $(SRCS:.cpp=.o)
OBJS        := $(OBJS:.cu=.o)

# =========================
# Build rules
# =========================
all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $^ -o $@ $(LIBS)

# Compile C++ (host code)
compile_attn_cuda.o: compile_attn_cuda.cpp flash_attn_cuda.h
	$(NVCC) -c $< -o $@ $(CXXFLAGS)

# Compile CUDA
flash_attn_cuda.o: flash_attn_cuda.cu flash_attn_cuda.h
	$(NVCC) -c $< -o $@ $(CUDAFLAGS)

clean:
	rm -f *.o $(TARGET)