# CPU source for big hdf5 input
SRC = $(wildcard source/*.c)
OBJ = ${SRC:.c=.o}
OUT = reconstructN_cpu

CC = gcc
CFLAGS = -O2 -std=gnu99 -lhdf5_hl -lhdf5 -lgsl -lgslcblas -lm -lz
INCLUDES = -I./include
DFLAGS = -DRECONSTRUCT_BACKWARDS -DMULTI_IMAGE_FILE

.PHONY: clean all linux

# Default target: uses h5cc to find HDF5 paths automatically
all: linux

# Linux target: Uses h5cc compiler wrapper to find HDF5 header and library paths
linux: CFLAGS += $(shell h5cc -show | grep -o -- "-I[^ ]*" 2>/dev/null || echo "") \
				$(shell h5cc -show | grep -o -- "-L[^ ]*" 2>/dev/null || echo "")
linux: $(OUT)

polaris: CFLAGS += -I/opt/cray/pe/hdf5/1.12.1.3/gnu/9.1/include \
				  -L/opt/cray/pe/hdf5/1.12.1.3/gnu/9.1/lib
polaris: $(OUT)

hpcs: CFLAGS += -I/clhome/aps_tools/shared/include \
				-L/clhome/aps_tools/shared/lib \
				-I/clhome/aps_tools/hdf5-1.8.2/hdf5/include \
				-L/clhome/aps_tools/hdf5-1.8.2/hdf5/lib
hpcs: $(OUT)

$(OUT): $(OBJ)
	mkdir -p bin
	@$(CC) $^ -o bin/$@ $(CFLAGS)
	@printf "\e[33mLinking\e[90m %s\e[0m\n" bin/$@
	@printf "\e[34mDone!\e[0m\n"

%.o: %.c
	@$(CC) $(DFLAGS) $(CFLAGS) $(INCLUDES) -c $< -o $@
	@printf "\e[36mCompile\e[90m %s\e[0m\n" $@

clean:
	@rm -f $(OBJ) *~ bin/$(OUT)
	@printf "\e[34mAll clear!\e[0m\n"
