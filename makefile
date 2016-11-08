# for cuda
# TODO: merge Rakefile used for src

VPATH = src

.PHONY: fmt TAGS

NVCC_FLAGS = -O2 -arch=sm_30 -g
CFLAGS = -O2 -std=c99 -Wall -Wextra

all: cpu cuda
cpu: cpumain
cuda: cumain device_prop cumulti cucomm cusingle

cumain: idas_parallel.cu
	nvcc -o $@ $(NVCC_FLAGS) $<

cusingle: idas.cu
	nvcc -o $@ $(NVCC_FLAGS) $<

cumulti: idas_multi.cu
	nvcc -o $@ $(NVCC_FLAGS) $<

cucomm: idas_comm_test.cu
	nvcc -o $@ $(NVCC_FLAGS) $<

cpumain: idas_cpu.c
	gcc -o $@ $(CFLAGS) $<

device_prop: device_props.cu
	nvcc -o $@ $(NVCC_FLAGS) $<

fmt: idas.cu idas_cpu.c
	clang-format -i $^

TAGS:
	ctags -R
