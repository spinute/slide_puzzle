# for cuda
# TODO: merge Rakefile used for src

VPATH = src

.PHONY: fmt TAGS 

CUDA_PATH = /usr/local/cuda
CUDA_INC += -I$(CUDA_PATH)/include
DEBUGFLAGS = -G -Xcompiler -rdynamic -gencode arch=compute_30,code=sm_30 -g
NVCC_FLAGS = -O2  -arch=sm_30 #$(DEBUGFLAGS)
CFLAGS = -O2 -std=c99 -Wall -Wextra

all: cpu cuda
cpu: cpumain cpu25
cuda: cumain device_prop cumulti cucomm cusingle

cumain: idas_distr.cu
	nvcc -o $@ $(NVCC_FLAGS) $<

idas_parallel.o: src/idas_parallel.cu
	nvcc --device-c $(NVCC_FLAGS) $<

.c.o:
	gcc $(CFLAGS) -c $< $(CUDA_INC)

cusingle: idas.cu
	nvcc -o $@ $(NVCC_FLAGS) $<

cumulti: idas_multi.cu
	nvcc -o $@ $(NVCC_FLAGS) $<

cucomm: idas_comm_test.cu
	nvcc -o $@ $(NVCC_FLAGS) $<

cpumain: idas_cpu.c
	gcc -o $@ $(CFLAGS) $<

cpu25: idas_cpu_25.c
	gcc -o $@ $(CFLAGS) $<

device_prop: device_props.cu
	nvcc -o $@ $(NVCC_FLAGS) $<

fmt: idas.cu idas_cpu.c
	clang-format -i $^

TAGS:
	ctags -R
