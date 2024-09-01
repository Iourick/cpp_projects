#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/tuple.h>
#include <thrust/host_vector.h>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

#include "kernel.cuh"
#include "Constants.h"


#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <thrust/sequence.h>

#include <thrust/gather.h>

#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

//------------------------------------------------------------------------------------------------
__global__
void calc_plan_and_values_for_regrouping_kernel(const Cand* d_arrCand
    , const  int* d_arrIndeces
    , const int QUantCand
    , const  int* d_arrGroupBeginIndecies
    , const int member_offset
    , const int& d_bin_metrics
    , int* d_arrValues
    , int* d_arrRegroupingPlan)
{
    extern __shared__ int  iarr[];
    const int QUantGroups = gridDim.x;
    const int numGroup = blockIdx.x;
    int iBeginGroupIndeces = d_arrGroupBeginIndecies[numGroup];
    int iEndGroupIndeces = (numGroup == (QUantGroups - 1)) ? QUantCand : d_arrGroupBeginIndecies[numGroup + 1];
    int lenGroup = iEndGroupIndeces - iBeginGroupIndeces;


    int idx = threadIdx.x;
    if (idx >= lenGroup)
    {
        iarr[idx] = 0;
        return;
    }


    // Get the value of the member using the offset
   // d_arrValues[indCur] = *reinterpret_cast<const int*>(base + member_offset);
    int quantSubGroups = 0;
    int stride = blockDim.x;
    for (int i = idx; i < lenGroup; i += stride)
    {
        int indCur = iBeginGroupIndeces + i;
        // extract value
        const char* base = reinterpret_cast<const char*>(&d_arrCand[d_arrIndeces[indCur]]);
        // Get the value of the member using the offset
        int ivalCur = *reinterpret_cast<const int*>(base + member_offset);
        d_arrValues[indCur] = ivalCur;
        if (0 == idx)
        {
            ++quantSubGroups;
            continue;
        }
        int indCurPrev = indCur - 1;
        // extract value
        const char* basePrev = reinterpret_cast<const char*>(&d_arrCand[d_arrIndeces[indCurPrev]]);
        // Get the value of the member using the offset
        int ivalCurPrev = *reinterpret_cast<const int*>(basePrev + member_offset);
        if ((ivalCur - ivalCurPrev) > d_bin_metrics)
        {
            ++quantSubGroups;
        }
    }
    iarr[threadIdx.x] = quantSubGroups;
    __syncthreads();

    // Parallel reduction within the block to sum partial sums
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            iarr[threadIdx.x] += iarr[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (0 == threadIdx.x)
    {
        d_arrRegroupingPlan[blockIdx.x] = iarr[0];
    }

}
void initializeOnGPU(const int QUantCand, const int QUantBeginIndeces, Cand** d_arrCand, int** d_arrIndeces, int** d_arrGroupBeginIndecies, int** d_arrValues, int** d_arrRegroupingPlan, int** d_bin_metrics) {
    // Initialize host vector with example data
    thrust::host_vector<Cand> h_vctCandHeap = {
        {10, 2, 3, 4.0f}, {26, 5, 6, 7.0f}, {12, 8, 9, 10.0f}, {50, 11, 12, 13.0f}, {28, 14, 15, 16.0f},
        {11, 3, 4, 5.0f}, {27, 6, 7, 8.0f}, {25, 9, 10, 11.0f}, {51, 12, 13, 14.0f}, {29, 15, 16, 17.0f},
        {52, 18, 19, 20.0f}, {70, 21, 22, 23.0f}
       , {121, 39, 40, 41.0f}, {122, 42, 43, 44.0f} , {80, 24, 25, 26.0f}, {90, 27, 28, 29.0f}, {91, 30, 31, 32.0f},
        {92, 33, 34, 35.0f}, {120, 36, 37, 38.0f}, {150, 45, 46, 47.0f}
    };

   /* thrust::host_vector<int> h_arrIndeces(QUantCand);
    thrust::sequence(h_arrIndeces.begin(), h_arrIndeces.end());
    std::random_shuffle(h_arrIndeces.begin(), h_arrIndeces.end());*/
    thrust::host_vector<int> h_arrIndeces = {0,5,2,7,1,6,4,9,3,8,10,11
    ,14,15,16,17,18,12,13,19};


    thrust::host_vector<int> h_arrGroupBeginIndecies(QUantBeginIndeces);
    h_arrGroupBeginIndecies[0] = 0;
    h_arrGroupBeginIndecies[1] = 12;

    // Allocate memory on the device
    cudaMalloc((void**)d_arrCand, QUantCand * sizeof(Cand));
    cudaMalloc((void**)d_arrIndeces, QUantCand * sizeof(int));
    cudaMalloc((void**)d_arrGroupBeginIndecies, QUantBeginIndeces * sizeof(int));
    cudaMalloc((void**)d_arrValues, QUantCand * sizeof(int));
    cudaMalloc((void**)d_arrRegroupingPlan, QUantBeginIndeces * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(*d_arrCand, thrust::raw_pointer_cast(h_vctCandHeap.data()), QUantCand * sizeof(Cand), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_arrIndeces, thrust::raw_pointer_cast(h_arrIndeces.data()), QUantCand * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_arrGroupBeginIndecies, thrust::raw_pointer_cast(h_arrGroupBeginIndecies.data()), QUantBeginIndeces * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize d_bin_metrics
    int h_bin_metrics = 2;
    cudaMalloc((void**)d_bin_metrics, sizeof(int));
    cudaMemcpy(*d_bin_metrics, &h_bin_metrics, sizeof(int), cudaMemcpyHostToDevice);
}
//-------------------------------------------------------------------------------------
void printMember(const Cand& cand, int member_offset) {
    const char* base = reinterpret_cast<const char*>(&cand);
    const int* member_ptr = reinterpret_cast<const int*>(base + member_offset);
    std::cout << *member_ptr;
}
//------------------------------------------------------------------------------------------
int main() {
    const int QUantCand = 20;
    const int QUantBeginIndeces = 2;

    Cand* d_arrCand;
    int* d_arrIndeces;
    int* d_arrGroupBeginIndecies;
    int* d_arrValues;
    int* d_arrRegroupingPlan;
    int* d_bin_metrics;

    // Initialize and allocate memory on GPU
    initializeOnGPU(QUantCand, QUantBeginIndeces, &d_arrCand, &d_arrIndeces, &d_arrGroupBeginIndecies, &d_arrValues, &d_arrRegroupingPlan, &d_bin_metrics);

    // Copy data back to host for printing
    thrust::host_vector<Cand> h_vctCandHeap(QUantCand);
    cudaMemcpy(thrust::raw_pointer_cast(h_vctCandHeap.data()), d_arrCand, QUantCand * sizeof(Cand), cudaMemcpyDeviceToHost);

    thrust::host_vector<int> h_arrIndeces(QUantCand);
    cudaMemcpy(thrust::raw_pointer_cast(h_arrIndeces.data()), d_arrIndeces, QUantCand * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::host_vector<int> h_arrGroupBeginIndecies(QUantBeginIndeces);
    cudaMemcpy(thrust::raw_pointer_cast(h_arrGroupBeginIndecies.data()), d_arrGroupBeginIndecies, QUantBeginIndeces * sizeof(int), cudaMemcpyDeviceToHost);

    int h_bin_metrics;
    cudaMemcpy(&h_bin_metrics, d_bin_metrics, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "d_arrCand:" << std::endl;
    for (int i = 0; i < QUantCand; ++i) {
        std::cout << i << " {" << h_vctCandHeap[i].mt << ", " << h_vctCandHeap[i].mdt << ", " << h_vctCandHeap[i].mwidth << ", " << h_vctCandHeap[i].msnr << "}" << std::endl;
    }

    std::cout << "d_arrIndeces: ";
    for (int i = 0; i < QUantCand; ++i) {
        std::cout << h_arrIndeces[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "d_arrGroupBeginIndecies: ";
    for (int i = 0; i < QUantBeginIndeces; ++i) {
        std::cout << h_arrGroupBeginIndecies[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "d_bin_metrics: " << h_bin_metrics << std::endl;

    
    int member_offset = offsetof(Cand, mt);
    std::cout << "/**********EXPECTED RESULTS****************/" << std::endl;
    std::cout << "             values:" << std::endl;
    std::cout << "First line: ";
    for (int i = 0; i < 12; ++i) {
        int index = h_arrIndeces[i];
        const Cand& cand = h_vctCandHeap[index];
        printMember(cand, member_offset);
        if (i < 11) std::cout << ", ";
    }
    std::cout << std::endl;

    // Print the second line (8 elements)
    std::cout << "Second line: ";
    for (int i = 12; i < QUantCand; ++i) {
        int index = h_arrIndeces[i];
        const Cand& cand = h_vctCandHeap[index];
        printMember(cand, member_offset);
        if (i < QUantCand - 1) std::cout << ", ";
    }
    std::cout << std::endl;
    std::cout << "d_arrRegroupingPlan =  {4,  4}" << std::endl;
    std::cout << "/************** RESULTS *******************************/" << std::endl;
/*---------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------*/
/*------------  END OF INITIALIZATION  ---------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------------------------------*/
    int threads_per_block = 1024;
    int blocks_per_grid = QUantBeginIndeces;

    calc_plan_and_values_for_regrouping_kernel<<< blocks_per_grid, threads_per_block, threads_per_block * sizeof(int)>>>
        (d_arrCand
        , d_arrIndeces
        , QUantCand
        , d_arrGroupBeginIndecies
        , member_offset
        , *d_bin_metrics
        , d_arrValues
        , d_arrRegroupingPlan);

    thrust::host_vector<int> h_arrValues(QUantCand);
    cudaMemcpy(thrust::raw_pointer_cast(h_arrValues.data()), d_arrValues, QUantCand * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::host_vector<int> h_arrRegroupingPlan(QUantBeginIndeces);
    cudaMemcpy(thrust::raw_pointer_cast(h_arrRegroupingPlan.data()), d_arrRegroupingPlan, QUantBeginIndeces * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "d_arrValues: ";
    for (int i = 0; i < QUantCand; ++i) {
        std::cout << h_arrValues[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "d_arrRegroupingPlan: ";
    for (int i = 0; i < QUantBeginIndeces; ++i) {
        std::cout << h_arrRegroupingPlan[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "d_bin_metrics: " << h_bin_metrics << std::endl;

    // Free allocated memory
    cudaFree(d_arrCand);
    cudaFree(d_arrIndeces);
    cudaFree(d_arrGroupBeginIndecies);
    cudaFree(d_arrValues);
    cudaFree(d_arrRegroupingPlan);
    cudaFree(d_bin_metrics);

    return 0;
}




//
//int main() 
//{
//    // 1 Create arrays  and stuff them on GPU
//    const int rows = 3;// 256;
//    const int cols = 16;// (1 << 20) / rows;
//    thrust::device_vector<fdmt_type_ >d_vctInp(rows * cols);
//    fdmt_type_* d_arr = thrust::raw_pointer_cast(d_vctInp.data());
//    
//    thrust::device_vector<fdmt_type_ >d_vctNorm(rows * cols);
//
//    fdmt_type_* d_norm = thrust::raw_pointer_cast(d_vctNorm.data());
//    
//    float* arr = (float*)malloc(rows * cols * sizeof(fdmt_type_));
//    float* norm = (float*)malloc(rows * cols * sizeof(fdmt_type_));
//    for (int i = 0; i < rows * cols; ++i)
//    {
//        arr[i] = 2;
//        norm[i] = 4.;
//    }
//    arr[4] = 140.;
//    arr[11] = 120.;
//    arr[2 * cols +4] = 100.;
//   
//    
//    for (int i = 3; i < 5; ++i)
//    {
//        //norm[cols + i] = 0.001;
//    }
//
//    cudaMemcpy(d_arr, arr, rows * cols * sizeof(fdmt_type_), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_norm, norm, rows * cols * sizeof(fdmt_type_), cudaMemcpyHostToDevice);
//    // !1
//
// 
//    // 4. metrics array on host and device
//    thrust::host_vector<int> h_bin_metrics(3);
//    h_bin_metrics[0] = 2;
//    h_bin_metrics[1] = 2;
//    h_bin_metrics[2] = 2;
//
//    thrust::device_vector<int> d_bin_metrics = h_bin_metrics;
//    // !4
//
//  
//
//  
///************* ! *******************************************************/
//
//   
//    
//
//    int n_repeat = 5;
//
//    /////////////////////////////////////////////////////////////////////////////
//    /////////////////////////////////////////////////////////////////////////////
//    ///////////  1 var  //////////////////////////////////////////////////////////////////
//    /////////////////////////////////////////////////////////////////////////////
//    auto start = std::chrono::high_resolution_clock::now();
//    for (int i = 0; i < n_repeat; ++i)
//    {
//        //if (doDetectSignal_v1(d_arr, d_norm, rows
//        //    , cols, WndWidth, valTresh, gridSize, blockSize, sz
//        //    , d_pAuxArray, d_pAuxIntArray, d_pAuxWidthArray, pstructOut))
//
//        //{
//        //    /*std::cout << "detection" << std::endl;
//        //    std::cout << "row = " << pstructOut->irow << std::endl;
//        //    std::cout << "col = " << pstructOut->icol << std::endl;
//        //    std::cout << "width = " << pstructOut->iwidth << std::endl;*/
//        //    //std::cout << "snr = " << pstructOut->snr << std::endl;
//        //}
//    }
//    auto end = std::chrono::high_resolution_clock::now();
//    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
//
//    std::cout << "doDetectSignal_v1 time:                     " << duration.count() / ((float)n_repeat) << " microseconds" << std::endl;
//   
//  
//
//    std::cout << "detect_signal_gpu time:                     " << duration.count() / ((float)n_repeat) << " microseconds" << std::endl;
//     
//    free(arr);
//    free(norm); 
//
//
//    return 0;
//}
//
