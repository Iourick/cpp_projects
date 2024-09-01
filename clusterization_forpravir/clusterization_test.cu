#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

#include <thrust/host_vector.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cuda_runtime.h>

#include "clusterization_test.cuh"
#include "Clusterization.cuh"
#include "Constants.h"


#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <thrust/sequence.h>

#include <thrust/gather.h>

#include <iostream>

#include <vector>
#include <algorithm>
#include "select_delegates.cuh"

#define MAX_CANDIDATE_PER_CHUNK 4096

using namespace clusterization;

//----------------------------------------------------------------------------------------------------------------
// CUDA kernel to initialize arrays
__global__ void initialize_arrays(int* arrt, int* arrdt, int* arrwidth, int* arrsnr, int num_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements)
    {
        arrt[idx] = num_elements - idx;
        arrdt[idx] = num_elements - idx;
        arrwidth[idx] = num_elements - idx;
        arrsnr[idx] = num_elements - idx;
    }
}
//----------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------
// CUDA kernel to initialize Cand vector
__global__ void initialize_cand(Cand* d_vec_cand, int* arrt, int* arrdt, int* arrwidth, int* arrsnr, int num_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements)
    {
        d_vec_cand[idx].mt = arrt[idx];
        d_vec_cand[idx].mdt = arrdt[idx];
        d_vec_cand[idx].mwidth = arrwidth[idx];
    }
}
//---------------------------------------------------------------------------------------
int main() 
{
    // 1 Create arrays  and stuff them on GPU
    int rows = 3;// 256;
    int cols = 16;// (1 << 20) / rows;
    thrust::device_vector<fdmt_type_ >d_vctInp(rows * cols);
    fdmt_type_* d_arr = thrust::raw_pointer_cast(d_vctInp.data());
    
    thrust::device_vector<fdmt_type_ >d_vctNorm(rows * cols);

    fdmt_type_* d_norm = thrust::raw_pointer_cast(d_vctNorm.data());
    
    float* arr = (float*)malloc(rows * cols * sizeof(fdmt_type_));
    float* norm = (float*)malloc(rows * cols * sizeof(fdmt_type_));
    for (int i = 0; i < rows * cols; ++i)
    {
        arr[i] = 2;
        norm[i] = 4.;
    }
    arr[4] = 140.;
    arr[11] = 120.;
    arr[2 * cols +4] = 100.;
   
    
    for (int i = 3; i < 5; ++i)
    {
     //  norm[cols + i] = 0.001;
    }

    cudaMemcpy(d_arr, arr, rows * cols * sizeof(fdmt_type_), cudaMemcpyHostToDevice);
    cudaMemcpy(d_norm, norm, rows * cols * sizeof(fdmt_type_), cudaMemcpyHostToDevice);
    // !1

    // 2.max length of window
    const int WndWidth = 2;
    //!2

    // 3. treshold
    float valTresh_ = 4.;
    float* d_pTresh;     // Device pointer
   

    // Allocate memory on the GPU for the float variable
    cudaError_t err = cudaMalloc((void**)&d_pTresh, sizeof(float));
    cudaMemcpy(d_pTresh, &valTresh_, sizeof(float), cudaMemcpyHostToDevice);
   
    // !3

    // 4. metrics array on host and device
    thrust::host_vector<int> h_bin_metrics(3);
    h_bin_metrics[0] = 2;
    h_bin_metrics[1] = 1;
    h_bin_metrics[2] = 2;

    thrust::device_vector<int> d_bin_metrics( h_bin_metrics.size());
    d_bin_metrics = h_bin_metrics;
    // !4

    // 5. normalization 
    //thrust::device_vector<float> d_normalized_fdmt(rows * cols);
    //const int blocksize0 = 1024;
    //const int gridsize0 = (rows * cols + blocksize0 - 1) / blocksize0;    
    //normalize_fdmt_kernel << <gridsize0, blocksize0 >> > (d_arr, d_norm, rows * cols, thrust::raw_pointer_cast(d_normalized_fdmt.data()));
    //
    //auto max_iter = thrust::max_element(d_normalized_fdmt.begin(), d_normalized_fdmt.end());
    //float max_value = *max_iter;


    //// 6. digitizing normalized fdmt

    //// Create the d_digitized vector
    //thrust::device_vector<int> d_digitized(d_normalized_fdmt.size());

    //// Apply the transformation to digitize the values
    //thrust::transform(d_normalized_fdmt.begin(), d_normalized_fdmt.end(),
    //    d_digitized.begin(), DigitizeFunctor(max_value));

   // thrust::device_vector<int> d_fdmt_digitized(rows * cols);
    const int* d_pbin_metrics = thrust::raw_pointer_cast(d_bin_metrics.data());

    // start of work
    const int blocksize0 = 1024;
    //clusterization::digitize_kernel << <1, blocksize0 , blocksize0  * sizeof(int) >> > (d_arr,  rows * cols);
   
   /* cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << "pipets" << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
    bool are_equal = thrust::equal(d_fdmt_digitized.begin(), d_fdmt_digitized.end(), d_digitized.begin());

    int iarr[48] = { 0 }, iarr1[48] = { 0 };
    cudaMemcpy(iarr1, thrust::raw_pointer_cast(d_fdmt_digitized.data()), 48 * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 48; ++i)
    {
        iarr[i] = (int)(arr[i] /sqrtf(0.0001 + norm[i]) * 128.0 / 70.0);
    }*/
    /*clusterization(const int* d_digitized_fdmt
        , const int Rows
        , const int Cols
        , const int iVAlTresh
        , const int WndWidth
        , const int* d_pbin_metrics
        , const std::string & filename)*/ 

    
    std::string filename = "ouput.log";
 
    clusterization::clusterization_main(d_arr
        , rows
        , cols
        , *d_pTresh
        , WndWidth
        , d_pbin_metrics
        , filename
        ,0.00001
       , 0.0
        );
  
    //-----
    Cand* d_arrCand = nullptr;

    cudaMalloc((void**)&d_arrCand, MAX_CANDIDATE_PER_CHUNK * sizeof(Cand));
    unsigned int* d_pimax_candidates_per_chunk = nullptr;
    cudaMalloc((void**)&d_pimax_candidates_per_chunk, sizeof(unsigned int));
    unsigned int itemp = MAX_CANDIDATE_PER_CHUNK;
    cudaMemcpy(d_pimax_candidates_per_chunk, &itemp, sizeof(unsigned int), cudaMemcpyHostToDevice);

    itemp = 0;
    unsigned int* d_pquantCand = 0;
    cudaMalloc((void**)&d_pquantCand, sizeof(unsigned int));
    cudaMemcpy(d_pquantCand, &itemp, sizeof(unsigned int), cudaMemcpyHostToDevice);
    const int QUantPower2_Wnd = 1;
    const dim3 blockSize = dim3(256, 1, 1);
    const dim3 gridSize = dim3((cols + blockSize.x - 1) / blockSize.x, rows, 1);
   clusterization::gather_candidates_in_fixedArray_kernel_v0<<<gridSize, blockSize >>>(d_arr, cols, *d_pTresh
        , QUantPower2_Wnd, d_pimax_candidates_per_chunk, d_arrCand, d_pquantCand[0]);


   Cand* d_arrCand1 = nullptr;
   unsigned int h_quantCand1 = 0;   

   clusterization::gather_candidates_in_dynamicalArray(d_arr, rows, cols
       , *d_pTresh, QUantPower2_Wnd, &d_arrCand1, &h_quantCand1);
   err = cudaGetLastError();
   if (err != cudaSuccess) {
       std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
       return 1;
   }  
   std::string filename1 = "ouput1.log";
   clusterization::write_log_v0(d_arrCand1
       , h_quantCand1
       , filename1
       , 0.00001
       , 0.0);
   
   err = cudaGetLastError();
   if (err != cudaSuccess) {
       std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
       return 1;
   }
   std::string  filename_csv1("Candidates1.csv");
   clusterization::writeCandDeviceArrayToCSV(d_arrCand1, h_quantCand1, filename_csv1);

   err = cudaGetLastError();
   if (err != cudaSuccess) {
       std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
       return 1;
   }

   // gather candidates with plan
   Cand* d_arrCand2 = nullptr;
   unsigned int* d_pquantCand2 = nullptr;
   itemp = 0;
   cudaMalloc((void**)&d_pquantCand2, sizeof(unsigned int));
   cudaMemcpy(d_pquantCand2, &itemp, sizeof(unsigned int), cudaMemcpyHostToDevice);
   err = cudaGetLastError();
   if (err != cudaSuccess) {
       std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
       return 1;
   }
   unsigned int h_quantCand2 = 0;
   clusterization::gather_candidates_in_heap_with_plan(d_arr, rows, cols
       , *d_pTresh
       , WndWidth
       , &d_arrCand2// = nullptr
       , &h_quantCand2
   );
   //unsigned int h_quantCand2 = 0;

   //cudaMemcpy(&h_quantCand2, d_pquantCand2, sizeof(unsigned int), cudaMemcpyDeviceToHost);
   std::string filename2 = "ouput2.log";
   clusterization::write_log_v0(d_arrCand2
       , h_quantCand2
       , filename2
       , 0.00001
       , 0.0);


   std::string  filename_csv2("Candidates2.csv");
   clusterization::writeCandDeviceArrayToCSV(d_arrCand2, h_quantCand2, filename_csv2);

    free(arr);
    free(norm); 
    cudaFree(d_arrCand);
    cudaFree(d_arrCand1);
    cudaFree(d_arrCand2);
    cudaFree(d_pquantCand2);

    return 0;
}

