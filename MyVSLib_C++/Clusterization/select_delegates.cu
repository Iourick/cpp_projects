#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/extrema.h>
#include <iostream>
#include <fstream>
#include <iomanip> // Include this header for std::setw
#include <algorithm>
#include <vector>
#include <random>
#include <numeric> // Include this header for std::iota
#include <chrono> // Include this header for timing
#include <cfloat>
#include "select_delegates.cuh"


    void select_delegates::select_delegates(const thrust::device_vector<Cand>& d_vctCandHeap,
        const thrust::device_vector<int>& d_vctIndices,
        const thrust::device_vector<int>& d_vctGroupBeginIndices,
        thrust::device_vector<int>& d_vctNumDelegates)
    {
        thrust::host_vector<int> h_vctGroupBeginIndices = d_vctGroupBeginIndices;
        thrust::host_vector<int> h_vctIndices = d_vctIndices;
        thrust::host_vector<int> h_vctNumDelegates(h_vctGroupBeginIndices.size());

        for (size_t i = 0; i < h_vctGroupBeginIndices.size() - 1; ++i)
        {
            int begin = h_vctGroupBeginIndices[i];
            int end = h_vctGroupBeginIndices[i + 1];

            CompareCand comp(thrust::raw_pointer_cast(d_vctCandHeap.data()));

            auto max_iter = thrust::max_element(
                d_vctIndices.begin() + begin, d_vctIndices.begin() + end, comp);

            h_vctNumDelegates[i] = (max_iter != d_vctIndices.end()) ? *max_iter : -1;
        }

        // Handle the last group
        int last_begin = h_vctGroupBeginIndices[h_vctGroupBeginIndices.size() - 1];
        int last_end = d_vctIndices.size();

        CompareCand comp(thrust::raw_pointer_cast(d_vctCandHeap.data()));

        auto max_iter = thrust::max_element(
            d_vctIndices.begin() + last_begin, d_vctIndices.begin() + last_end, comp);

        h_vctNumDelegates[h_vctGroupBeginIndices.size() - 1] = (max_iter != d_vctIndices.end()) ? *max_iter : -1;

        // Copy result back to device
        d_vctNumDelegates = h_vctNumDelegates;
    }
    //---------------------------------------------------------
// sample for call
//int threads_per_block = 256;
//int blocks_per_grid = d_vctGroupBeginIndeces.size();
//select_delegates_kernel << < blocks_per_grid, threads_per_block, threads_per_block* (sizeof(int) + sizeof(float)) >> >
//(thrust::raw_pointer_cast(d_vctCandHeap.data()),
//    thrust::raw_pointer_cast(d_vctIndeces.data()),
//    d_vctCandHeap.size(),
//    thrust::raw_pointer_cast(d_vctGroupBeginIndeces.data()),
//    thrust::raw_pointer_cast(d_vctNumDelegates0.data()));
    __global__
        void select_delegates::select_delegates_kernel(const Cand* d_arrCandHeap,
            int* d_arrIndices,
            const int QUantCand,
            int* d_arrGroupBeginIndeces,
            int* d_arrNumDelegates)
    {
        extern __shared__ float  arr[];
        int* nums = (int*)(arr + blockDim.x);
        //------
        const int QUantGroups = gridDim.x;
        const int numGroup = blockIdx.x;

        int iBeginGroupIndeces = d_arrGroupBeginIndeces[numGroup];
        int iEndGroupIndeces = (numGroup == (QUantGroups - 1)) ? QUantCand : d_arrGroupBeginIndeces[numGroup + 1];
        int lenGroup = iEndGroupIndeces - iBeginGroupIndeces;

        int idx = threadIdx.x;
        float val = -FLT_MAX;
        int numCur = -1;
        if (idx >= lenGroup)
        {

            val = -FLT_MAX;
            numCur = idx;
        }
        else
        {
            for (int i = idx; i < lenGroup; i += blockDim.x)
            {
                int itemp = d_arrIndices[iBeginGroupIndeces + i];
                float temp = d_arrCandHeap[itemp].msnr;
                if (temp > val)
                {
                    val = temp;
                    numCur = itemp;
                }

            }
        }
        arr[idx] = val;
        nums[idx] = numCur;
        __syncthreads();

        // Parallel reduction within the block to sum partial sums
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
        {
            if (threadIdx.x < s)
            {
                if (arr[threadIdx.x + s] > arr[threadIdx.x])
                {
                    arr[threadIdx.x] = arr[threadIdx.x + s];
                    nums[threadIdx.x] = nums[threadIdx.x + s];
                }

            }
            __syncthreads();
        }
        if (0 == threadIdx.x)
        {
            d_arrNumDelegates[blockIdx.x] = nums[0];
        }
        __syncthreads();
    }
    //----------------------------------------------------------------------------------------------
    void select_delegates::write_log(const Cand * d_parrCand,
        const int QUantCandidates,
        const thrust::device_vector<int>& d_vctGroupBeginIndices,
        const thrust::device_vector<int>& d_vctIndices,
        const thrust::device_vector<int>& d_vctNumDelegates,
        const std::string& filename
        , const float delt_bin
        , const float t_chunkBegin)
    {

        // Copy device vectors to host vectors
        //thrust::host_vector<Cand> h_vctCandHeap = d_vctCandHeap;
        Cand* h_parrCand = (Cand*)malloc(QUantCandidates * sizeof(Cand));
        cudaMemcpy(h_parrCand, d_parrCand, QUantCandidates * sizeof(Cand), cudaMemcpyDeviceToHost);
        
        thrust::host_vector<int> h_vctGroupBeginIndices = d_vctGroupBeginIndices;
        thrust::host_vector<int> h_vctIndices = d_vctIndices;
        thrust::host_vector<int> h_vctNumDelegates = d_vctNumDelegates;

        // Open log file
        std::ofstream logFile(filename);

        // Write header
        logFile << std::setw(10) << "N"
            << std::setw(10) << "t, bin"
            << std::setw(10) << "t, sec"
            << std::setw(10) << "dedisp"
            << std::setw(10) << "width"
            << std::setw(10) << "SNR"
            << std::setw(10) << "num" << std::endl;

        // Write data
        for (size_t i = 0; i < h_vctNumDelegates.size(); ++i) {
            int j = h_vctNumDelegates[i];
            Cand cand = h_parrCand[j];
            int num = (i == h_vctGroupBeginIndices.size() - 1)
                ? h_vctIndices.size() - h_vctGroupBeginIndices[i]
                : h_vctGroupBeginIndices[i + 1] - h_vctGroupBeginIndices[i];
            logFile << std::setw(10) << i + 1
                << std::setw(10) << cand.mt
                << std::setw(10) << t_chunkBegin + cand.mt * delt_bin
                << std::setw(10) << cand.mdt
                << std::setw(10) << cand.mwidth
                << std::setw(10) << cand.msnr
                << std::setw(10) << num << std::endl;
        }

        // Close log file
        logFile.close();
        free(h_parrCand);
    }

/*-----------------------------------------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------------------------------------*/
void print_log(const thrust::device_vector<Cand>& d_vctCandHeap,
    const thrust::device_vector<int>& d_vctGroupBeginIndices,
    const thrust::device_vector<int>& d_vctIndices,
    const thrust::device_vector<int>& d_vctNumDelegates) 
{

    // Copy device vectors to host vectors
    thrust::host_vector<Cand> h_vctCandHeap = d_vctCandHeap;
    thrust::host_vector<int> h_vctGroupBeginIndices = d_vctGroupBeginIndices;
    thrust::host_vector<int> h_vctIndices = d_vctIndices;
    thrust::host_vector<int> h_vctNumDelegates = d_vctNumDelegates;

    // Print h_vctNumDelegates
    std::cout << "vctNumDelegates: " << std::endl;
    for (size_t i = 0; i < h_vctNumDelegates.size(); ++i)
    {
        std::cout << " " << h_vctNumDelegates[i];
    }
    std::cout << std::endl << std::endl;

    // Print header
    std::cout << std::setw(10) << "N"
        << std::setw(10) << "time"
        << std::setw(10) << "dedisp"
        << std::setw(10) << "width"
        << std::setw(10) << "SNR"
        << std::setw(10) << "num" << std::endl;

    // Print data
    for (size_t i = 0; i < h_vctNumDelegates.size(); ++i)
    {
        int j = h_vctNumDelegates[i];
        Cand cand = h_vctCandHeap[j];
        int num = (i == h_vctGroupBeginIndices.size() - 1)
            ? h_vctIndices.size() - h_vctGroupBeginIndices[i]
            : h_vctGroupBeginIndices[i + 1] - h_vctGroupBeginIndices[i];
        std::cout << std::setw(10) << i
            << std::setw(10) << cand.mt
            << std::setw(10) << cand.mdt
            << std::setw(10) << cand.mwidth
            << std::setw(10) << cand.msnr
            << std::setw(10) << num << std::endl;
    }
}



