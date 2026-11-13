# Project GAIA — Filler Kernel Build
# Usage: make          (auto-detect arch)
#        make ARCH=sm_110   (force Thor)

ARCH ?= sm_110
NVCC ?= nvcc
NVCC_FLAGS = -O3 -arch=$(ARCH)

TARGETS = filler_kernel

all: $(TARGETS)

filler_kernel: filler_kernel.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@ -lcuda

clean:
	rm -f $(TARGETS)

.PHONY: all clean
