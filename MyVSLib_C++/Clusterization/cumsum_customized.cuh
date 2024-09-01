
#ifndef CUMSUM_CUSTOMIZED_CUH
#define CUMSUM_CUSTOMIZED_CUH

__global__
void calc_part_cumsum_kernel(const int* d_arrRegroupingPlan, const int LEnarr, int* d_arr_cumsum);

__global__
void calc_part_cumsum_kernel_v1(const int* d_arrRegroupingPlan, const int LEnarr, int* d_arr_cumsum);

void calc_cumsum_custom_gpu(int* d_arrRegroupingPlan
    , int* d_arr_buff, const int LEnarr, const int treads_per_block);

void fnc_cumsum_customized_gpu(int* d_arrInput
    , int* d_arr_buff, const int LEnarr, int* d_arrOutput);

__global__
void calc_cumsum_kernel_v0(const int* d_arrRegroupingPlan, const int LEnarr, int* d_arr_cumsum);

__global__
void calc_part_cumsum_kernel_new(const int* d_arrRegroupingPlan, const int LEnarr, int* d_arr_cumsum, unsigned long long clock_rate);

void fnc_cumsum_customized_gpu_new(int* d_arrInput
    , int* d_arr_buff, const int LEnarr, int* d_arrOutput,
    unsigned long long clock_rate);

__global__
void scanKernel(int* d_out, int* d_in, int size);

void inclusiveScan_CUB(float* d_arr, const int LEn);

void inclusiveScan_CUB(int* d_arr, const int LEn);
#endif 