VPATH = src

.PHONY: fmt TAGS clean

CUDA_PATH = /usr/local/cuda
CUDA_INC += -I$(CUDA_PATH)/include
NVCC_FLAGS = --compiler-options -Wall -arch=sm_61  --resource-usage -Xcompiler -rdynamic -G -gencode arch=compute_61,code=sm_61 -g -lineinfo
#NVCC_FLAGS = -Xcompiler -O2 -Xptxas -O2 --compiler-options -Wall -arch=sm_61  --resource-usage
CFLAGS = -O2 -std=c99 -Wall -Wextra
objects = cumain custatic device_prop cusingle cubase cpumain cpu25

all: cpu cuda
cuda: cumain cublack custatic device_prop cusingle cubase cudynamic
cpu: cpumain cpu25

cumain: idas_global.cu
	nvcc -o $@ $(NVCC_FLAGS) $<
cudynamic: idas_dynamic.cu
	nvcc -o $@ $(NVCC_FLAGS) $<
cublack: idas_black.cu
	nvcc -o $@ $(NVCC_FLAGS) $<
custatic: idas_static.cu
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
