# CPU source for big hdf5 input - THREADED VERSION
SRC = $(wildcard source/*.c)
OBJ = ${SRC:.c=.o}
OUT = reconstructN_cpu

# Add OpenMP support
CFLAGS = -O2 -std=gnu99 -fopenmp
LDFLAGS = -lhdf5_hl -lhdf5 -lgsl -lgslcblas -lm -lz -fopenmp
RPATH_FLAGS = -Wl,-rpath,'$$ORIGIN/../../../lib' -Wl,-rpath,'$(CONDA_PREFIX)/lib'
CC = gcc
INCLUDES = -I./include
DFLAGS = -DRECONSTRUCT_BACKWARDS -DMULTI_IMAGE_FILE

# Find h5cc - prefer conda environment version
ifdef CONDA_PREFIX
    H5CC := $(shell if [ -x "$(CONDA_PREFIX)/bin/h5cc" ]; then echo "$(CONDA_PREFIX)/bin/h5cc"; else echo "h5cc"; fi)
else
    H5CC := h5cc
endif

.PHONY: clean all linux

# Default target
all: linux

# Linux target with OpenMP
linux: CFLAGS += $(shell $(H5CC) -show 2>/dev/null | grep -o -- "-I[^ ]*" || echo "")
linux: LDFLAGS := $(shell $(H5CC) -show 2>/dev/null | grep -o -- "-L[^ ]*" || echo "") \
                  $(LDFLAGS)
linux: $(OUT)

# Polaris target (update for OpenMP)
polaris: CFLAGS += -I/opt/cray/pe/hdf5/1.12.1.3/gnu/9.1/include -fopenmp
polaris: LDFLAGS := -L/opt/cray/pe/hdf5/1.12.1.3/gnu/9.1/lib \
                    $(LDFLAGS)
polaris: $(OUT)

hpcs: CFLAGS += -I/clhome/aps_tools/shared/include \
                -I/clhome/aps_tools/hdf5-1.8.2/hdf5/include
hpcs: LDFLAGS := -L/clhome/aps_tools/shared/lib \
                -L/clhome/aps_tools/hdf5-1.8.2/hdf5/lib \
                $(LDFLAGS)
hpcs: $(OUT)

$(OUT): $(OBJ)
	mkdir -p bin
	@$(CC) $^ -o bin/$@ $(LDFLAGS) $(RPATH_FLAGS)
	@printf "\e[33mLinking\e[90m %s\e[0m\n" bin/$@
	@printf "\e[34mDone!\e[0m\n"

%.o: %.c
	@$(CC) $(DFLAGS) $(CFLAGS) $(INCLUDES) -c $< -o $@
	@printf "\e[36mCompile\e[90m %s\e[0m\n" $@

clean:
	@rm -f $(OBJ) *~ bin/$(OUT)
	@printf "\e[34mAll clear!\e[0m\n"
