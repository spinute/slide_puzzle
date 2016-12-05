VPATH = src

.PHONY: fmt TAGS clean

CUDA_PATH = /usr/local/cuda
CUDA_INC += -I$(CUDA_PATH)/include
DEBUGFLAGS = -G -Xcompiler -rdynamic -gencode arch=compute_30,code=sm_30 -g
NVCC_FLAGS = -O2  -arch=sm_30 #$(DEBUGFLAGS)
CFLAGS = -O2 -std=c99 -Wall -Wextra
objects = cumain device_prop cusingle cubase cpumain cpu25

all: cpu cuda
cuda: cumain device_prop cusingle cubase
cpu: cpumain cpu25

cumain: idas_distr.cu
	nvcc -o $@ $(NVCC_FLAGS) $<
cubase: idas_distr_baseline.cu
	nvcc -o $@ $(NVCC_FLAGS) $<
cusingle: idas.cu
	nvcc -o $@ $(NVCC_FLAGS) $<
device_prop: device_props.cu
	nvcc -o $@ $(NVCC_FLAGS) $<

cpumain: idas_cpu.c
	gcc -o $@ $(CFLAGS) $<
cpu25: idas_cpu_25.c
	gcc -o $@ $(CFLAGS) $<

clean :
	@-rm $(objects)
fmt: idas.cu idas_cpu.c
	clang-format -i $^
TAGS:
	ctags -R
