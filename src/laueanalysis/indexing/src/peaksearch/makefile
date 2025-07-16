SRC = $(wildcard src/*.c)
OBJ = ${SRC:.c=.o}
OUT = peaksearch

# Separate compile flags from link flags
CFLAGS = -O2 -std=gnu99 -msse2
LDFLAGS = -lhdf5_hl -lhdf5 -lgsl -lgslcblas -lm -lz
CC = gcc

# Find h5cc - prefer conda environment version
ifdef CONDA_PREFIX
    H5CC := $(shell if [ -x "$(CONDA_PREFIX)/bin/h5cc" ]; then echo "$(CONDA_PREFIX)/bin/h5cc"; else echo "h5cc"; fi)
else
    H5CC := h5cc
endif

# Linux target: Uses h5cc compiler wrapper to find HDF5 header and library paths
linux: CFLAGS += $(shell $(H5CC) -show 2>/dev/null | grep -o -- "-I[^ ]*" || echo "")
linux: LDFLAGS := $(shell $(H5CC) -show 2>/dev/null | grep -o -- "-L[^ ]*" || echo "") \
				$(LDFLAGS)
linux: $(OUT)

polaris: CFLAGS += -I/opt/cray/pe/hdf5/1.12.1.3/gnu/9.1/include
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
	@$(CC) $^ -o $@ $(LDFLAGS)
	@printf "\e[33mLinking\e[90m %s\e[0m\n" $@
	@printf "\e[34mDone!\e[0m\n"

%.o: %.c
	@$(CC) -c $< -o $@ $(CFLAGS)
	@printf "\e[36mCompile\e[90m %s\e[0m\n" $@

clean:
	@rm -f $(OUT) $(OBJ)
	@printf "\e[34mAll clear!\e[0m\n"
