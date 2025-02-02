
#include "cudaLib.cuh"
#include <cstdint>

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

__global__ void saxpy_gpu(float *x, float *y, float scale, int size) {
  //	Insert GPU SAXPY kernel code here
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size)
    y[i] += scale * x[i];
}

int runGpuSaxpy(int vectorSize) {

  std::cout << "Hello GPU Saxpy!\n";

  uint64_t size = vectorSize * sizeof(float);

  float *x, *y, *y_init;
  float *x_d, *y_d;

  x = (float *)malloc(size);
  y = (float *)malloc(size);
  y_init = (float *)malloc(size);

  if (x == NULL || y == NULL || y_init == NULL) {
    printf("runGpuSaxpy: unable to malloc memory");
    return -1;
  }

  vectorInit(x, vectorSize);
  vectorInit(y, vectorSize);

  std::memcpy(y_init, y, size);
  float scale = 2.7f;

  cudaError_t err;
  err = cudaMalloc((void **)&x_d, size);
  gpuErrchk(err);

  err = cudaMalloc((void **)&y_d, size);
  gpuErrchk(err);

  err = cudaMemcpy(x_d, x, size, cudaMemcpyHostToDevice);
  gpuErrchk(err);

  err = cudaMemcpy(y_d, y, size, cudaMemcpyHostToDevice);
  gpuErrchk(err);

  // Launch kernel
  // 256 threads in a TB
  saxpy_gpu<<<(vectorSize + 255) / 256, 256>>>(x_d, y_d, scale, vectorSize);

  err = cudaMemcpy(y, y_d, size, cudaMemcpyDeviceToHost);
  gpuErrchk(err);

  int error_count = verifyVector(x, y_init, y, scale, vectorSize);
  std::cout << "Found " << error_count << " / " << vectorSize << " errors \n";

  cudaFree(x_d);
  cudaFree(y_d);
  free(x);
  free(y);

  return 0;
}

/*
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is
 responsible for sampleSize points. *pSums is a pointer to an array that holds
 the number of 'hit' points for each thread. The length of this array is
 pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__ void generatePoints(uint64_t *pSums, uint64_t pSumSize,
                               uint64_t sampleSize) {

  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  if (tid >= pSumSize)
    return;

  float x, y;
  uint64_t hit_count = 0;

  curandState_t rng;
  curand_init(clock64(), tid, 0, &rng);

  for (int i = 0; i < sampleSize; ++i) {
    x = curand_uniform(&rng);
    y = curand_uniform(&rng);

    if (x * x + y * y <= 1.0f) {
      ++hit_count;
    }
  }

  pSums[tid] = hit_count;
}

__global__ void reduceCounts(uint64_t *pSums, uint64_t *totals,
                             uint64_t pSumSize, uint64_t reduceSize) {

  int tid = threadIdx.x + blockDim.x * blockIdx.x;


  if (tid >= pSumSize / reduceSize)
    return;


  uint64_t reduced_sum = 0;
  for (int i = 0; i < reduceSize; ++i) {
    // assert((reduceSize * tid + i) < pSumSize);
    reduced_sum += pSums[reduceSize * tid + i];
  }

  totals[tid] = reduced_sum;
}

int runGpuMCPi(uint64_t generateThreadCount, uint64_t sampleSize,
               uint64_t reduceThreadCount, uint64_t reduceSize) {

  //  Check CUDA device presence
  int numDev;
  cudaGetDeviceCount(&numDev);
  if (numDev < 1) {
    std::cout << "CUDA device missing!\n";
    return -1;
  }

  auto tStart = std::chrono::high_resolution_clock::now();

  float approxPi = estimatePi(generateThreadCount, sampleSize,
                              reduceThreadCount, reduceSize);

  std::cout << std::setprecision(10);
  std::cout << "Estimated Pi = " << approxPi << "\n";

  auto tEnd = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> time_span = (tEnd - tStart);
  std::cout << "It took " << time_span.count() << " seconds.";

  return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize,
                  uint64_t reduceThreadCount, uint64_t reduceSize) {

  double approxPi = 0;
  uint64_t total_hits = 0;

  std::vector<uint64_t> totals(reduceThreadCount);

  uint64_t *pSums_d, *totals_d;
  cudaError_t err;

  err = cudaMalloc((void **)&pSums_d, generateThreadCount * sizeof(uint64_t));
  gpuErrchk(err);

  err = cudaMalloc((void **)&totals_d, reduceThreadCount * sizeof(uint64_t));
  gpuErrchk(err);

  generatePoints<<<(generateThreadCount + 255) / 256, 256>>>(
      pSums_d, generateThreadCount, sampleSize);

  reduceCounts<<<(reduceThreadCount + 255) / 256, 256>>>(
      pSums_d, totals_d, generateThreadCount, reduceSize);

  err =
      cudaMemcpy(totals.data(), totals_d, sizeof(uint64_t) * reduceThreadCount,
                 cudaMemcpyDeviceToHost);

  gpuErrchk(err);

  for (const auto &total : totals) {
    total_hits += total;
  }

  approxPi = (((double)total_hits / sampleSize) / generateThreadCount) * 4.0f;

  cudaFree(pSums_d);
  cudaFree(totals_d);

  return approxPi;
}
