OUT = liblaue.so
PIXELS2QS_SRC = ../pixels2qs/src
PEAKSEARCH_SRC = ../peaksearch/src
EULER_SRC = ../euler/src
SOURCES = liblaue.c \
	$(PIXELS2QS_SRC)/readGeoN.c \
	$(PIXELS2QS_SRC)/keyLists.c \
	$(PIXELS2QS_SRC)/checkFileType.c \
	$(PIXELS2QS_SRC)/xmlUtility.c \
	$(PIXELS2QS_SRC)/main.c \
	$(PEAKSEARCH_SRC)/peaksearch.c \
	$(PEAKSEARCH_SRC)/grid.c \
	$(PEAKSEARCH_SRC)/grid_operations.c \
	$(PEAKSEARCH_SRC)/fitToFunction.c \
	$(PEAKSEARCH_SRC)/fitToGaussian.c \
	$(PEAKSEARCH_SRC)/point.c \
	$(PEAKSEARCH_SRC)/list.c \
	$(PEAKSEARCH_SRC)/peak.c \
	$(PEAKSEARCH_SRC)/genfileinf.c \
	$(PEAKSEARCH_SRC)/calibparam.c \
	$(PEAKSEARCH_SRC)/ccdTable.c \
	$(EULER_SRC)/fast.c \
	$(EULER_SRC)/EulerTestFaster.c \
	$(EULER_SRC)/lattice.c \
	$(EULER_SRC)/mathUtil.c \
	$(EULER_SRC)/optimize.c

CC ?= gcc
H5CC ?= h5cc
CFLAGS ?= -O2
CFLAGS += -fPIC -flto -ffunction-sections -fdata-sections -fvisibility=hidden -std=gnu99 -msse2 -DLIBLAUE_BUILD \
	-I$(PIXELS2QS_SRC) -I$(PEAKSEARCH_SRC) -I$(EULER_SRC) \
	$(shell $(H5CC) -show 2>/dev/null | grep -o -- "-I[^ ]*" || true)
LDFLAGS += -shared -flto -Wl,--gc-sections -lgsl -lgslcblas -lm

$(OUT): $(SOURCES) liblaue.h
	$(CC) $(CFLAGS) $(SOURCES) -o $@ $(LDFLAGS)

clean:
	rm -f $(OUT)
