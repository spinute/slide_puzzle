# for cuda
# TODO: merge Rakefile used for src

VPATH = src

.PHONY: fmt TAGS

CUDA_PATH = /usr/local/cuda
CUDA_INC = += -I$(CUDA_PATH)/include
DEBUGFLAGS = -G -Xcompiler -rdynamic -gencode arch=compute_30,code=sm_30
NVCC_FLAGS = -O2  -arch=sm_30 #-g $(DEBUGFLAGS)
CFLAGS = -O2 -std=c99 -Wall -Wextra

all: cpu cuda
cpu: cpumain cpu25
cuda: cumain device_prop cumulti cucomm cusingle

# FIXME: .c file can be compiled separately, and then linked with cuda codes

%.o : %.cu
	nvcc $(NVCC_FLAGS) -o $@ -c $< $(CUDA_INC)

cumain: idas_parallel.cu distributor.c queue.c ht.c state.c utils.c
	nvcc -dc $(NVCC_FLAGS) $<
	gcc -c src/distributor.c src/queue.c src/ht.c src/state.c src/utils.c
	gcc -L$(CUDA_PATH)/lib64 idas_parallel.o distributor.o queue.o ht.o state.o utils.o -lcudart

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
