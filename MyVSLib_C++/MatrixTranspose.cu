#include "MatrixTranspose.cuh"



__global__ void transpose(float* input, float* output, int width, int height)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // Shared memory tile

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Transpose data from global to shared memory
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }

    __syncthreads();

    // Calculate new indices for writing to output
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // Transpose data from shared to global memory
    if (x < height && y < width) {
        output[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}

