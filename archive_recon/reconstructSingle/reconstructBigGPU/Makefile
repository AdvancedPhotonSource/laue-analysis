#Makefile for cuda file
HDF5_BASE = "/clhome/KYUE/lib/hdf5"
GSL_BASE = "/clhome/KYUE/lib/gsl"
 
BIN      := reconstructN

NVCC ?= nvcc
INCD = -I${HDF5_BASE}/include -I./include -I${GSL_BASE}/include
LIBS = -L${HDF5_BASE}/lib -L${GSL_BASE}/lib -lhdf5_hl -lhdf5 -lgsl -lgslcblas -lm -lz -lpthread -ldl -lcudadevrt
NVCCFLAGS := -arch=sm_20 --ptxas-options=-v 
DFLAGS = -DRECONSTRUCT_BACKWARDS -DMULTI_IMAGE_FILE

# files
CPP_SOURCES       := $(wildcard source/*.c)
CU_SOURCES        := source/WireScan.cu
HEADERS           := $(wildcard include/*.h)
CPP_OBJS          := $(patsubst %.c, %.o, $(CPP_SOURCES))
CU_OBJS           := $(patsubst %.cu, %.o, $(CU_SOURCES))

%.o: %.c
	$(NVCC) $(DFLAGS) -c  $(INCD) -o $@ $< $(LIBS)
	
%.o : %.cu
	$(NVCC) $(DFLAGS) -c $(NVCCFLAGS) $(INCD)  -o $@ $< $(LIBS)

$(BIN): $(CPP_OBJS) $(CU_OBJS)
	$(NVCC) $(DFLAGS) -o bin/$(BIN) $(CU_OBJS) $(CPP_OBJS) $(INCD) $(LIBS)

clean:
	rm -f bin/$(BIN) source/*.o 

depend: $(CPP_SOURCES) $(CU_SOURCES)
	makedepend $(INCD) $^
