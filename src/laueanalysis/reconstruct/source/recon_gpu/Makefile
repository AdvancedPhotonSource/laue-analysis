# Makefile for CUDA reconstruction with multiple HDF5 input
# Updated to support conda environments and modern CUDA architectures

# Binary name
BIN := reconstructN_gpu

# Build mode (debug or release)
MODE ?= release

# GPU Architecture (default to V100, override with make GPU_ARCH=sm_80 for A100)
GPU_ARCH ?= sm_70

# Detect conda environment
ifdef CONDA_PREFIX
    HDF5_BASE = $(CONDA_PREFIX)
    GSL_BASE = $(CONDA_PREFIX)
    # Try to find CUDA in conda first, then system
    NVCC := $(shell if [ -x "$(CONDA_PREFIX)/bin/nvcc" ]; then echo "$(CONDA_PREFIX)/bin/nvcc"; else echo "nvcc"; fi)
    # Use h5cc for HDF5 flags if available
    H5CC := $(shell if [ -x "$(CONDA_PREFIX)/bin/h5cc" ]; then echo "$(CONDA_PREFIX)/bin/h5cc"; else echo "h5cc"; fi)
else
    # Fallback to system paths
    HDF5_BASE = /usr/local
    GSL_BASE = /usr/local
    NVCC = nvcc
    H5CC = h5cc
endif

# Extract HDF5 flags dynamically
HDF5_CFLAGS := $(shell $(H5CC) -show 2>/dev/null | grep -o -- "-I[^ ]*" || echo "-I$(HDF5_BASE)/include")
HDF5_LDFLAGS := $(shell $(H5CC) -show 2>/dev/null | grep -o -- "-L[^ ]*" || echo "-L$(HDF5_BASE)/lib")

# Include directories
INCD = $(HDF5_CFLAGS) -I./include -I$(GSL_BASE)/include

# Libraries
LIBS = $(HDF5_LDFLAGS) -L$(GSL_BASE)/lib -lhdf5_hl -lhdf5 -lgsl -lgslcblas -lm
# Check if zlib is available and add it
ZLIB_CHECK := $(shell pkg-config --exists zlib 2>/dev/null && echo "yes")
ifeq ($(ZLIB_CHECK),yes)
    LIBS += $(shell pkg-config --libs zlib)
else
    LIBS += -lz
endif

# Runtime library paths
RPATH_FLAGS = -Xlinker -rpath -Xlinker '$$ORIGIN' -Xlinker -rpath -Xlinker '$(CONDA_PREFIX)/lib'

# Compiler flags
CXXFLAGS := -g
NVCCFLAGS := -arch=$(GPU_ARCH) --ptxas-options=-v

# Set flags based on build mode
ifeq ($(MODE),debug)
    NVCCFLAGS += -G -g -O0
    CXXFLAGS += -O0
else
    NVCCFLAGS += -O3
    CXXFLAGS += -O2
endif

# Source files
CPP_SOURCES := $(wildcard source/*.c)
CU_SOURCES := source/WireScan.cu
HEADERS := $(wildcard include/*.h)
CPP_OBJS := $(patsubst %.c, %.o, $(CPP_SOURCES))
CU_OBJS := $(patsubst %.cu, %.o, $(CU_SOURCES))

# Default target
.PHONY: all
all: bin $(BIN)

# Create bin directory
bin:
	@mkdir -p bin

# Compile C files
%.o: %.c
	@echo "Compiling C file: $<"
	@$(NVCC) -c $(INCD) -o $@ $< $(LIBS)

# Compile CUDA files
%.o: %.cu
	@echo "Compiling CUDA file: $<"
	@$(NVCC) -c $(NVCCFLAGS) $(INCD) -o $@ $< $(LIBS)

# Link everything
$(BIN): $(CPP_OBJS) $(CU_OBJS)
	@echo "Linking: bin/$(BIN)"
	@$(NVCC) -o bin/$(BIN) $(CU_OBJS) $(CPP_OBJS) $(INCD) $(LIBS) $(RPATH_FLAGS)
	@echo "Build complete: bin/$(BIN)"

# Clean build artifacts
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	@rm -f bin/$(BIN) source/*.o
	@echo "Clean complete"

# Show configuration
.PHONY: show-config
show-config:
	@echo "=== Build Configuration ==="
	@echo "MODE: $(MODE)"
	@echo "NVCC: $(NVCC)"
	@echo "HDF5_BASE: $(HDF5_BASE)"
	@echo "GSL_BASE: $(GSL_BASE)"
	@echo "GPU_ARCH: $(GPU_ARCH)"
	@echo "HDF5_CFLAGS: $(HDF5_CFLAGS)"
	@echo "HDF5_LDFLAGS: $(HDF5_LDFLAGS)"
	@echo "CONDA_PREFIX: $(CONDA_PREFIX)"
	@echo "=========================="

# Test compilation without linking
.PHONY: test-compile
test-compile:
	@echo "Testing CUDA compilation..."
	@$(NVCC) --version
	@echo "Testing includes..."
	@$(NVCC) -c $(NVCCFLAGS) $(INCD) -o /tmp/test.o source/WireScan.cu --dry-run

# Generate dependencies
.PHONY: depend
depend: $(CPP_SOURCES) $(CU_SOURCES)
	@echo "Generating dependencies..."
	@makedepend $(INCD) $^ 2>/dev/null || true

# Help target
.PHONY: help
help:
	@echo "GPU Reconstruction Makefile"
	@echo ""
	@echo "Usage: make [target] [options]"
	@echo ""
	@echo "Targets:"
	@echo "  all          - Build the executable (default)"
	@echo "  clean        - Remove build artifacts"
	@echo "  show-config  - Display build configuration"
	@echo "  test-compile - Test compilation setup"
	@echo "  depend       - Generate dependencies"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Options:"
	@echo "  MODE=debug|release - Set build mode (default: release)"
	@echo "  GPU_ARCH=sm_XX     - Set GPU architecture (default: sm_70)"
	@echo ""
	@echo "Examples:"
	@echo "  make                    # Build in release mode"
	@echo "  make MODE=debug         # Build in debug mode"
	@echo "  make GPU_ARCH=sm_80     # Build for A100 GPUs"
	@echo "  make clean              # Clean build artifacts"
	@echo "  make show-config        # Show current configuration"
