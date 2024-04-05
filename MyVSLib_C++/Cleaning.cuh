#include "cuda_runtime.h"
void cleanInpFDMT_v0(float* d_arr, const int NRows, const int NCols, float* d_buff);

__global__ void calcRowDisps_kernel(float* d_arrIm, const int nRows, const int nCols, float* arrDisps);

__global__
void 
calculateMeanAndSTD_kernel(float* d_arr, const unsigned int len, float* pmean, float* pstd);

__global__ 
void clean_out_the_trash_kernel_v2(float* d_arr, const int NRows, const int NCols, float* d_buff, const float mean, const float std);