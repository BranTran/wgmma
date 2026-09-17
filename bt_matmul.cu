#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda/barrier>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <ctime>
#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <unistd.h>

//typedef __nv_bfloat16 bf16;
typedef uint16_t bf16;

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(1);
  }
}
#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

// One wgmma.mma_async.sync.aligned.m64nNk16 shape per N (N must be a
// compile-time constant baked into the instruction mnemonic, so each valid N
// - any multiple of 8 in [8, 256] - gets its own generated header; see
// examples/bt/generate_matmul_kernel.py).
#include "examples/bt/matmul_8.cuh"
#include "examples/bt/matmul_16.cuh"
#include "examples/bt/matmul_24.cuh"
#include "examples/bt/matmul_32.cuh"
#include "examples/bt/matmul_40.cuh"
#include "examples/bt/matmul_48.cuh"
#include "examples/bt/matmul_56.cuh"
#include "examples/bt/matmul_64.cuh"
#include "examples/bt/matmul_72.cuh"
#include "examples/bt/matmul_80.cuh"
#include "examples/bt/matmul_88.cuh"
#include "examples/bt/matmul_96.cuh"
#include "examples/bt/matmul_104.cuh"
#include "examples/bt/matmul_112.cuh"
#include "examples/bt/matmul_120.cuh"
#include "examples/bt/matmul_128.cuh"
#include "examples/bt/matmul_136.cuh"
#include "examples/bt/matmul_144.cuh"
#include "examples/bt/matmul_152.cuh"
#include "examples/bt/matmul_160.cuh"
#include "examples/bt/matmul_168.cuh"
#include "examples/bt/matmul_176.cuh"
#include "examples/bt/matmul_184.cuh"
#include "examples/bt/matmul_192.cuh"
#include "examples/bt/matmul_200.cuh"
#include "examples/bt/matmul_208.cuh"
#include "examples/bt/matmul_216.cuh"
#include "examples/bt/matmul_224.cuh"
#include "examples/bt/matmul_232.cuh"
#include "examples/bt/matmul_240.cuh"
#include "examples/bt/matmul_248.cuh"
#include "examples/bt/matmul_256.cuh"
//#include "examples/matmul/matmul_1.cuh"
//#include "examples/matmul/matmul_2.cuh"
//#include "examples/matmul/matmul_3.cuh"
//#include "examples/matmul/matmul_4.cuh"
//#include "examples/matmul/matmul_5.cuh"
//#include "examples/matmul/matmul_6.cuh"
//#include "examples/matmul/matmul_7.cuh"
//#include "examples/matmul/matmul_8.cuh"
//#include "examples/matmul/matmul_9.cuh"
//#include "examples/matmul/matmul_10.cuh"
//#include "examples/matmul/matmul_11.cuh"
//#include "examples/matmul/matmul_12.cuh"

std::default_random_engine generator(69);
cublasHandle_t cublas_handle;
void runCublasGemmBF16(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C) {
  float alpha = 1, beta = 0;
  // C(column major) = A(row major) * B(column major)
  cublasStatus_t status = cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, A, CUDA_R_16BF,
    N, B, CUDA_R_16BF, K, &beta, C, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "CUBLAS error: " << status << std::endl;
    exit(1);
  }
}

void run_kernel(int kernel_num, int M, int N, int K, bf16 *A, bf16 *B, bf16 *C, unsigned long long iterations, int *DB = nullptr) {
  switch (kernel_num) {
    case 0:
      runCublasGemmBF16(M, N, K, A, B, C);
      break;
    case 8:
      runKernel8(M, N, K, A, B, C, DB, iterations);
      break;
    case 16:
      runKernel16(M, N, K, A, B, C, DB, iterations);
      break;
    case 24:
      runKernel24(M, N, K, A, B, C, DB, iterations);
      break;
    case 32:
      runKernel32(M, N, K, A, B, C, DB, iterations);
      break;
    case 40:
      runKernel40(M, N, K, A, B, C, DB, iterations);
      break;
    case 48:
      runKernel48(M, N, K, A, B, C, DB, iterations);
      break;
    case 56:
      runKernel56(M, N, K, A, B, C, DB, iterations);
      break;
    case 64:
      runKernel64(M, N, K, A, B, C, DB, iterations);
      break;
    case 72:
      runKernel72(M, N, K, A, B, C, DB, iterations);
      break;
    case 80:
      runKernel80(M, N, K, A, B, C, DB, iterations);
      break;
    case 88:
      runKernel88(M, N, K, A, B, C, DB, iterations);
      break;
    case 96:
      runKernel96(M, N, K, A, B, C, DB, iterations);
      break;
    case 104:
      runKernel104(M, N, K, A, B, C, DB, iterations);
      break;
    case 112:
      runKernel112(M, N, K, A, B, C, DB, iterations);
      break;
    case 120:
      runKernel120(M, N, K, A, B, C, DB, iterations);
      break;
    case 128:
      runKernel128(M, N, K, A, B, C, DB, iterations);
      break;
    case 136:
      runKernel136(M, N, K, A, B, C, DB, iterations);
      break;
    case 144:
      runKernel144(M, N, K, A, B, C, DB, iterations);
      break;
    case 152:
      runKernel152(M, N, K, A, B, C, DB, iterations);
      break;
    case 160:
      runKernel160(M, N, K, A, B, C, DB, iterations);
      break;
    case 168:
      runKernel168(M, N, K, A, B, C, DB, iterations);
      break;
    case 176:
      runKernel176(M, N, K, A, B, C, DB, iterations);
      break;
    case 184:
      runKernel184(M, N, K, A, B, C, DB, iterations);
      break;
    case 192:
      runKernel192(M, N, K, A, B, C, DB, iterations);
      break;
    case 200:
      runKernel200(M, N, K, A, B, C, DB, iterations);
      break;
    case 208:
      runKernel208(M, N, K, A, B, C, DB, iterations);
      break;
    case 216:
      runKernel216(M, N, K, A, B, C, DB, iterations);
      break;
    case 224:
      runKernel224(M, N, K, A, B, C, DB, iterations);
      break;
    case 232:
      runKernel232(M, N, K, A, B, C, DB, iterations);
      break;
    case 240:
      runKernel240(M, N, K, A, B, C, DB, iterations);
      break;
    case 248:
      runKernel248(M, N, K, A, B, C, DB, iterations);
      break;
    case 256:
      runKernel256(M, N, K, A, B, C, DB, iterations);
      break;
  }
}
int yo = 0;
void randomize_matrix(bf16 *mat, int N) {
  std::normal_distribution<float> distribution(0, 1);
  for (int i = 0; i < N; i++) {
    mat[i] = distribution(generator);
  }
  ++yo;
}

bool verify_matrix(bf16 *matRef, bf16 *matOut, int N) {
  double diff = 0.0;
  int i;
  for (i = 0; i < N; i++) {
    int r = i / 8192, c = i % 8192;
    int it = c*8192+r;
    diff = std::fabs(__bfloat162float(matRef[i] - matOut[i]));
    if (diff > 0.1) {
      printf("Divergence! Should %5.2f, Is %5.2f (Diff %5.2f) at %d\n",
      __bfloat162float(matRef[i]), __bfloat162float(matOut[i]), diff, i);
      return false;
    }
  }
  return true;
}

__global__ void warmupKernel() {
  __shared__ int s[100];
  s[0] += s[1];
}

int main(int argc, char** argv) {
 unsigned long long iterations;
 int kernel_num = 128; // Default kernel number
if (argc != 2 && argc != 3) {
    // Usage message now reflects the two accepted patterns
    fprintf(stderr, "usage: %s <iterations>\n", argv[0]);
    fprintf(stderr, "   or: %s <kernel_number> <iterations>\n", argv[0]);
    fprintf(stderr, "kernel_number: 128 (default), or any wgmma N shape (multiple of 8, 8-256)\n");
    exit(1);
} else if (argc == 3) {
    // Case 1: Two arguments provided
    kernel_num = atoi(argv[1]);
    iterations = atoll(argv[2]);
    printf("Running kernel %d for %lld iterations.\n", kernel_num, iterations);
} else { // argc == 2
    // Case 2: Only one argument provided
    iterations = atoll(argv[1]);
    printf("Running default kernel %d for %lld iterations.\n", kernel_num, iterations);
} 
  //warmupKernel<<<1024, 1024>>>();

  cublasCreate(&cublas_handle);
  float elapsed_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  long max_size = 8192;
  long m = max_size, n = max_size, k = max_size;

  bf16 *A = nullptr, *B = nullptr, *C = nullptr,
        *C_ref = nullptr;  // host matrices
  bf16 *dA = nullptr, *dB = nullptr, *dC = nullptr,
        *dC_ref = nullptr; // device matrices
  
  int *DB = nullptr; int *dDB = nullptr;  

  A = (bf16 *)malloc(sizeof(bf16) * max_size * max_size);
  B = (bf16 *)malloc(sizeof(bf16) * max_size * max_size);
  C = (bf16 *)malloc(sizeof(bf16) * max_size * max_size);
  C_ref = (bf16 *)malloc(sizeof(bf16) * max_size * max_size);
  DB = (int *)malloc(sizeof(int) * max_size * 128);
  cudaCheck(cudaMalloc((void **)&dDB, sizeof(int) * max_size * 128));

  randomize_matrix(A, max_size * max_size);
  randomize_matrix(B, max_size * max_size);
  randomize_matrix(C, max_size * max_size);
  
  cudaCheck(cudaMalloc((void **)&dA, sizeof(bf16) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dB, sizeof(bf16) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dC, sizeof(bf16) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dC_ref, sizeof(bf16) * max_size * max_size));
  
  cudaCheck(cudaMemcpy(dA, A, sizeof(bf16) * max_size * max_size,
  cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dB, B, sizeof(bf16) * max_size * max_size,
      cudaMemcpyHostToDevice));

  int repeat_times = 1;
  bool run_verif = false;
//  for (int kernel_num : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}) {
    // for (int kernel_num : {0, 11}) {
    // Give the GPU some rest to avoid thermal throttling
    //sleep(5);
    std::cout << "KERNEL " << kernel_num << std::endl;
    // Verify against cuBLAS. Also serves as a warmup step.
    if (run_verif) {
      memset(C, 0, sizeof(bf16) * max_size * max_size);
      cudaCheck(cudaMemcpy(dC, C, sizeof(bf16) * max_size * max_size, cudaMemcpyHostToDevice));
      cudaCheck(cudaMemcpy(dC_ref, C, sizeof(bf16) * max_size * max_size, cudaMemcpyHostToDevice));
      memset(DB, ~0, sizeof(int) * max_size * 128);
      cudaCheck(cudaMemcpy(dDB, DB, sizeof(int) * max_size * 128,
        cudaMemcpyHostToDevice));
      //run_kernel(0, m, n, k, dA, dB, dC_ref); // cuBLAS
      run_kernel(kernel_num, m, n, k, dA, dB, dC, iterations, dDB); // Executes the kernel, modifies the result matrix
      cudaCheck(cudaDeviceSynchronize());
      cudaCheck(cudaGetLastError()); // Check for async errors during kernel run
      cudaMemcpy(C, dC, sizeof(bf16) * max_size * max_size, cudaMemcpyDeviceToHost);
      cudaMemcpy(C_ref, dC_ref, sizeof(bf16) * max_size * max_size, cudaMemcpyDeviceToHost);

      if (kernel_num > 1 && !verify_matrix(C_ref, C, m * n)) {
        std::cout << "~~~~~~~~~~~~~~~~ Failed to pass the correctness verification against cuBLAS. ~~~~~~~~~~~~~~~~" << std::endl;
        printf("%f\n", __bfloat162float(C_ref[m]));
      }

      cudaMemcpy(DB, dDB, sizeof(int) * max_size * 8, cudaMemcpyDeviceToHost);

      int i = 0;
      long sumLoad = 0, cntLoad = 0;
      long sumCompute = 0, cntCompute = 0;
      long sumStore = 0, cntStore = 0;
      int times = 0;
      while (DB[i] != ~0) {
        sumLoad += DB[i], cntLoad += DB[i + 1];
        sumCompute += DB[i + 2], cntCompute += DB[i + 3];
        sumStore += DB[i + 4], cntStore += DB[i + 5];
        i += 6;
        times++;
      }
      if (times > 0) {
        printf("Load: %f, Compute: %f,  Store: %f, Datapoints: %d\n", (sumLoad + .0) / cntLoad, (sumCompute + .0) / cntCompute, (sumStore + .0) / cntStore, times);
      }

    }//if(run_verif)

    // Benchmark
    cudaEventRecord(start);
    for (int j = 0; j < repeat_times; j++) {
      run_kernel(kernel_num, m, n, k, dA, dB, dC, iterations);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("gpu execution time = %.3f ms\n", elapsed_time);  
   // long flops = (2LL * m) * (n * k);
   // printf(
   //     "Average elapsed time: (%7.6f) s, performance: (%7.1f) TFLOPS. size: (%ld).\n\n",
   //     elapsed_time / 1000.0 / repeat_times,
   //     (repeat_times * flops * 1e-9) / elapsed_time, m);
  //} //for loop

  // Free up CPU and GPU space
  free(A);
  free(B);
  free(C);
  free(C_ref);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaFree(dC_ref);
  return 0;
};
