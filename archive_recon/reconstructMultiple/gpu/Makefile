#Makefile for cuda file with multiple hdf input
#atomic add to float computation all on float 
#HDF5_BASE = "/local/kyue/hdf5"
#GSL_BASE = "/local/kyue/gsl"
HDF5_BASE = "/clhome/aps_tools/cudaRecon/lib/hdf5"
GSL_BASE = "/clhome/aps_tools/cudaRecon/lib/gsl"

BIN      := reconstructN
CXXFLAGS := -g

NVCC ?= /usr/local/cuda-6.5/bin/nvcc
INCD = -I${HDF5_BASE}/include -I./include -I${GSL_BASE}/include
LIBS = -L${HDF5_BASE}/lib -L${GSL_BASE}/lib -lhdf5_hl -lhdf5 -lgsl -lgslcblas -lm -lz
NVCCFLAGS := -arch=sm_20 --ptxas-options=-v -O3 -G -g 


# files
CPP_SOURCES       := $(wildcard source/*.c)
CU_SOURCES        := source/WireScan.cu
HEADERS           := $(wildcard include/*.h)
CPP_OBJS          := $(patsubst %.c, %.o, $(CPP_SOURCES))
CU_OBJS           := $(patsubst %.cu, %.o, $(CU_SOURCES))

%.o: %.c
	$(NVCC) -c  $(INCD)  -o $@ $< $(LIBS)
	
%.o : %.cu
	$(NVCC) -c $(NVCCFLAGS) $(INCD)  -o $@ $< $(LIBS)

$(BIN): $(CPP_OBJS) $(CU_OBJS)
	$(NVCC) -o bin/$(BIN) $(CU_OBJS) $(CPP_OBJS) $(INCD) $(LIBS)

clean:
	rm -f bin/$(BIN) source/*.o 

depend: $(CPP_SOURCES) $(CU_SOURCES)
	makedepend $(INCD) $^
