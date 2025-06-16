# CPU source for big hdf5 input
HDF5_BASE = "/clhome/KYUE/lib/hdf5"
GSL_BASE = "/clhome/KYUE/lib/gsl"

CC = gcc

CFLAGS = -g -fgnu89-inline -std=gnu99 -msse2
#CFLAGS =-fgnu89-inline -std=gnu99 -msse2 -fassociative-math -O3 -ftree-vectorize -march=native

INCLUDES = -I${HDF5_BASE}/include -I./include -I${GSL_BASE}/include

LFLAGS = -L${HDF5_BASE}/lib -L${GSL_BASE}/lib

DFLAGS = -DRECONSTRUCT_BACKWARDS -DMULTI_IMAGE_FILE

LIBS = -lhdf5_hl -lhdf5 -lgsl -lgslcblas -lm -lz

SRCS = $(wildcard source/*.c)

OBJS = $(SRCS:.c=.o)

OUT = reconstructN

.PHONY: depend clean

all: $(OUT)

$(OUT): $(OBJS)
	$(CC) $(DFLAGS) $(CFLAGS) $(INCLUDES) -o bin/$(OUT) $(OBJS) $(LFLAGS) $(LIBS)

.c.o:
	$(CC) $(DFLAGS) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean:
	$(RM) source/*.o *~ bin/$(OUT)

depend: $(SRCS)
	makedepend $(INCLUDES) $^
