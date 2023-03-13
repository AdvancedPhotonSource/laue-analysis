SRC = $(wildcard src/*.c)
OBJ = ${SRC:.c=.o}
OUT = pix2qs

CFLAGS = -O2 -std=gnu99	-lhdf5_hl -lhdf5 -lgsl -lgslcblas -msse2 -lm
CC = gcc


polaris: CFLAGS += -I/opt/cray/pe/hdf5/1.12.1.3/gnu/9.1/include \
				  -L/opt/cray/pe/hdf5/1.12.1.3/gnu/9.1/lib 

polaris: $(OUT)


hpcs: CFLAGS += -I/clhome/aps_tools/shared/include \
				-L/clhome/aps_tools/shared/lib \
				-I/clhome/aps_tools/hdf5-1.8.2/hdf5/include \
				-L/clhome/aps_tools/hdf5-1.8.2/hdf5/lib  \
				-lm -lz
hpcs: $(OUT)

$(OUT): $(OBJ)
	@$(CC) $^ -o $@ $(CFLAGS)
	@printf "\e[33mLinking\e[90m %s\e[0m\n" $@
	@printf "\e[34mDone!\e[0m\n"

%.o: %.c
	@$(CC) -c $< -o $@ $(CFLAGS)
	@printf "\e[36mCompile\e[90m %s\e[0m\n" $@ 

clean:
	@rm -f $(OUT) $(OBJ)
	@printf "\e[34mAll clear!\e[0m\n"