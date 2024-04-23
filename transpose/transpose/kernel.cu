
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#define TILE_DIM 32
#define  BLOCK_ROWS 8

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
#include <iostream>
#include <cuda_runtime.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8

#include <iostream>
#include <cuda_runtime.h>

#define TILE_DIM 32
#define BLOCK_ROWS 32//8

// Kernel function for coalesced matrix transposition
//__global__ void transposeCoalesced(float* odata, float* idata, int width, int height) {
//    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // Adding 1 to avoid bank conflicts
//
//    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
//    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
//    int index_in = xIndex + (yIndex)*width;
//
//    xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
//    yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
//    int index_out = xIndex + (yIndex)*height;
//
//    // Load data into shared memory
//    for (int i = 0; i < TILE_DIM && yIndex + i < height; i += blockDim.y) {
//        if (xIndex < width)
//            tile[threadIdx.y + i][threadIdx.x] = idata[index_in + i * width];
//    }
//
//    __syncthreads();
//
//    // Store data from shared memory to global memory
//    for (int i = 0; i < TILE_DIM && xIndex + i < height; i += blockDim.y) {
//        if (yIndex < width)
//            odata[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
//    }
//}
//------------------------------------------------------------
#define BLOCK_DIM 32//16
//dim3 grid((width + BLOCK_DIM - 1) / BLOCK_DIM, (height + BLOCK_DIM - 1) / BLOCK_DIM, 1);
//dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);
__global__ void transpose(float* odata, float* idata, int width, int height)
{
    __shared__ float block[BLOCK_DIM][BLOCK_DIM + 1];

    // read the matrix tile into shared memory
        // load one element per thread from device memory (idata) and store it
        // in transposed order in block[][]
    unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
    unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
    if ((xIndex < width) && (yIndex < height))
    {
        unsigned int index_in = yIndex * width + xIndex;
        block[threadIdx.y][threadIdx.x] = idata[index_in];
    }

    // synchronise to ensure all writes to block[][] have completed
    __syncthreads();

    // write the transposed matrix tile to global memory (odata) in linear order
    xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
    yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
    if ((xIndex < height) && (yIndex < width))
    {
        unsigned int index_out = yIndex * height + xIndex;
        odata[index_out] = block[threadIdx.x][threadIdx.y];
    }
}

//------------------------------------------
__global__
void transposeCoalesced_(float* output, float* input, const int width, const int height)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // Shared memory tile
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * gridDim.y + threadIdx.y;


    // Transpose data from global to shared memory
    if (x < width && y < height)
    {  
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
       printf(" 1  threadIdx.x = %i  threadIdx.y = %i  x = %i  y = %i  tile[threadIdx.y][threadIdx.x] = %f\n", threadIdx.x, threadIdx.y, x, y, tile[threadIdx.y][threadIdx.x]);
    }
    

    __syncthreads();
    // Calculate new indices for writing to output
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // Transpose data from shared to global memory
    if (x < height && y < width) {
        output[y * height + x] = tile[threadIdx.x][threadIdx.y];
        printf("threadIdx.x = %i  threadIdx.y = %i  x = %i  y = %i  output[y][x] = %f\n", threadIdx.x, threadIdx.y, x, y, output[y * height + x]);

    }

 
}

int main() {
    // Matrix dimensions
    int width = 33;
    int height = 2;

    // Host matrices
    float* h_inputMatrix = new float[width * height];
    float* h_outputMatrix = new float[height * width];

    // Initialize input matrix (you may fill this with actual data)
    for (int i = 0; i < width * height; ++i) {
        h_inputMatrix[i] = i;
    }

    // Device matrices
    float* d_inputMatrix;
    float* d_outputMatrix;
    cudaMalloc((void**)&d_inputMatrix, width * height * sizeof(float));
    cudaMalloc((void**)&d_outputMatrix, height * width * sizeof(float));

    // Copy input matrix to device
    cudaMemcpy(d_inputMatrix, h_inputMatrix, width * height * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 threadsPerBlock(TILE_DIM,2/*BLOCK_ROWS*/);
    dim3 numBlocks((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);

    // Launch kernel
   // transposeCoalesced_ << <numBlocks, threadsPerBlock >> > (d_outputMatrix, d_inputMatrix, width, height);
    dim3 grid((width + BLOCK_DIM -1) / BLOCK_DIM,( height + BLOCK_DIM -1) / BLOCK_DIM, 1);
    dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);
    transpose << <grid, threads >> > (d_outputMatrix, d_inputMatrix, width, height);

    // Copy result back to host
    cudaMemcpy(h_outputMatrix, d_outputMatrix, height * width * sizeof(float), cudaMemcpyDeviceToHost);

    // Output the result
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout << h_outputMatrix[i * width + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free device memory
    cudaFree(d_inputMatrix);
    cudaFree(d_outputMatrix);

    // Free host memory
    delete[] h_inputMatrix;
    delete[] h_outputMatrix;

    return 0;
}


// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
