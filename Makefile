CCCL_INCLUDE= -I$(SCRATCH)/cccl/libcudacxx/include
NVCC_FLAGS = -std=c++17 -O3 -DNDEBUG -w
OUT_DIR = out

# A standard CUDA Toolkit install keeps libcublas.so alongside libcudart in
# $(CUDA_HOME)/lib64, which nvcc already searches by default. NVIDIA HPC SDK
# installs instead split the math libraries out into a sibling
# math_libs/<ver>/.../lib(64) tree (sometimes only as a REDIST copy built
# against a different CUDA version than $(CUDA_HOME)), so -lcublas doesn't
# resolve there unless we locate and add that directory explicitly. This finds
# nothing (and leaves NVCC_LDFLAGS untouched) on a normal CUDA Toolkit install.
CUBLAS_LIBDIR := $(shell root=$$(dirname $$(dirname "$(CUDA_HOME)")); \
    hit=$$(find "$$root/math_libs" -name 'libcublas.so' 2>/dev/null | head -1); \
    [ -z "$$hit" ] && hit=$$(find "$$root" -name 'libcublas.so' 2>/dev/null | head -1); \
    [ -n "$$hit" ] && dirname "$$hit")

NVCC_LDFLAGS :=
# -L must precede -lcublas on the command line for the linker to use it.
ifneq ($(strip $(CUBLAS_LIBDIR)),)
NVCC_LDFLAGS += -L$(CUBLAS_LIBDIR) -Xlinker -rpath -Xlinker $(CUBLAS_LIBDIR)
endif
NVCC_LDFLAGS += -lcublas -lcuda

CUDA_OUTPUT_FILE = -o $(OUT_DIR)/$@
NCU_PATH := $(shell which ncu)
NCU_COMMAND = sudo $(NCU_PATH) --set full --import-source yes

NVCC_FLAGS += --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math -Xcompiler=-fPIE -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing $(CCCL_INCLUDE)
NVCC_FLAGS += -arch=sm_90a

NVCC_BASE = nvcc $(NVCC_FLAGS) $(NVCC_LDFLAGS) -lineinfo

sum: sum.cu 
	$(NVCC_BASE) $^ $(CUDA_OUTPUT_FILE)

sumprofile: sum
	$(NCU_COMMAND) -o $@ -f $(OUT_DIR)/$^

bt: bt_matmul.cu 
	mkdir -p $(OUT_DIR)
	$(NVCC_BASE) $^ -o $(OUT_DIR)/wgmma 

matmul: matmul.cu 
	mkdir -p $(OUT_DIR)
	$(NVCC_BASE) $^ $(CUDA_OUTPUT_FILE)

matmulprofile: matmul
	$(NCU_COMMAND) -o $@ -f $(OUT_DIR)/$^

clean:
	rm $(OUT_DIR)/*
