SRC = $(wildcard src/*.c)
OBJ = ${SRC:.c=.o}
OUT = pix2qs

CFLAGS = -O2 -lm -std=gnu99
CC = gcc

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