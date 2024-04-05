
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define TILE_DIM 32
__global__ void transpose(float* input, float* output, int width, int height);

__global__ void transpose(cufftComplex* input, cufftComplex* output, int width, int height);