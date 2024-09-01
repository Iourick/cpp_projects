#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include <iostream>
#include <fstream>
#include <iomanip> // Include this header for std::setw
#include <algorithm>
#include <vector>
#include <random>
#include <numeric> // Include this header for std::iota
#include <chrono> // Include this header for timing
#include <cfloat>

#define PROFILING true

struct SCand {
    int mt;
    int mdt;
    int mwidth;
    float msnr;
};
using Cand = SCand;

struct CompareCand
{
    const Cand* data;

    CompareCand(const Cand* data) : data(data) {}

    __device__ bool operator()(int a, int b) const
    {
        return data[a].msnr < data[b].msnr;
    }
};

struct CompareCandMember {
    const Cand* d_vctCandHeap;
    int member_offset;

    CompareCandMember(const Cand* _d_vctCandHeap, int _member_offset)
        : d_vctCandHeap(_d_vctCandHeap), member_offset(_member_offset) {}

    __host__ __device__
        bool operator()(const int& idx1, const int& idx2) const {
        const char* base1 = reinterpret_cast<const char*>(&d_vctCandHeap[idx1]);
        const char* base2 = reinterpret_cast<const char*>(&d_vctCandHeap[idx2]);
        int value1 = *reinterpret_cast<const int*>(base1 + member_offset);
        int value2 = *reinterpret_cast<const int*>(base2 + member_offset);

        // Print debug information within the functor
      //  printf("Comparing idx1: %d, value1: %d, idx2: %d, value2: %d\n", idx1, value1, idx2, value2);

        return value1 < value2;
    }
};

void select_delegates(const thrust::device_vector<Cand>& d_vctCandHeap,
    const thrust::device_vector<int>& d_vctIndices,
    const thrust::device_vector<int>& d_vctGroupBeginIndices,
    thrust::device_vector<int>& d_vctNumDelegates)
{
    thrust::host_vector<int> h_vctGroupBeginIndices = d_vctGroupBeginIndices;
    thrust::host_vector<int> h_vctIndices = d_vctIndices;
    thrust::host_vector<int> h_vctNumDelegates(h_vctGroupBeginIndices.size());

    for (size_t i = 0; i < h_vctGroupBeginIndices.size() - 1; ++i) {
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
//--------------------------------------------------------------------
void print_log(const thrust::device_vector<Cand>& d_vctCandHeap,
    const thrust::device_vector<int>& d_vctGroupBeginIndices,
    const thrust::device_vector<int>& d_vctIndices,
    const thrust::device_vector<int>& d_vctNumDelegates) {

    // Copy device vectors to host vectors
    thrust::host_vector<Cand> h_vctCandHeap = d_vctCandHeap;
    thrust::host_vector<int> h_vctGroupBeginIndices = d_vctGroupBeginIndices;
    thrust::host_vector<int> h_vctIndices = d_vctIndices;
    thrust::host_vector<int> h_vctNumDelegates = d_vctNumDelegates;

    // Print h_vctNumDelegates
    std::cout << "vctNumDelegates: " << std::endl;
    for (size_t i = 0; i < h_vctNumDelegates.size(); ++i) {
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
    for (size_t i = 0; i < h_vctNumDelegates.size(); ++i) {
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
void write_log(const thrust::device_vector<Cand>& d_vctCandHeap,
    const thrust::device_vector<int>& d_vctGroupBeginIndices,
    const thrust::device_vector<int>& d_vctIndices,
    const thrust::device_vector<int>& d_vctNumDelegates,
    const std::string& filename) {

    // Copy device vectors to host vectors
    thrust::host_vector<Cand> h_vctCandHeap = d_vctCandHeap;
    thrust::host_vector<int> h_vctGroupBeginIndices = d_vctGroupBeginIndices;
    thrust::host_vector<int> h_vctIndices = d_vctIndices;
    thrust::host_vector<int> h_vctNumDelegates = d_vctNumDelegates;

    // Open log file
    std::ofstream logFile(filename);

    // Write header
    logFile << std::setw(10) << "N"
        << std::setw(10) << "time"
        << std::setw(10) << "dedisp"
        << std::setw(10) << "width"
        << std::setw(10) << "SNR"
        << std::setw(10) << "num" << std::endl;

    // Write data
    for (size_t i = 0; i < h_vctNumDelegates.size(); ++i) {
        int j = h_vctNumDelegates[i];
        Cand cand = h_vctCandHeap[j];
        int num = (i == h_vctGroupBeginIndices.size() - 1)
            ? h_vctIndices.size() - h_vctGroupBeginIndices[i]
            : h_vctGroupBeginIndices[i + 1] - h_vctGroupBeginIndices[i];
        logFile << std::setw(10) << i +1
            << std::setw(10) << cand.mt
            << std::setw(10) << cand.mdt
            << std::setw(10) << cand.mwidth
            << std::setw(10) << cand.msnr
            << std::setw(10) << num << std::endl;
    }

    // Close log file
    logFile.close();
}


//---------------------------------------------------------
__global__
void select_delegates_kernel(Cand* d_arrCandHeap,
   int* d_arrIndices,
    const int QUantCand,
   int* d_arrGroupBeginIndeces,   
   int*d_arrNumDelegates)
{
    extern __shared__ float  arr[];
    int* nums = (int*)(arr + blockDim.x);
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
   /* if (0 == numGroup)
    {
        printf("iBeginGroupIndeces = %i  iEndGroupIndeces = %i  idx = %i  arr[idx] = %f  nums[idx] = %i\n", iBeginGroupIndeces, iEndGroupIndeces, idx, arr[idx], nums[idx]);
    }*/
    __syncthreads();

    // Parallel reduction within the block to sum partial sums
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            if (arr[threadIdx.x + s] > arr[threadIdx.x])
            {
                arr[threadIdx.x] = arr[threadIdx.x + s];
                nums[threadIdx.x] =nums[threadIdx.x + s];
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
//--------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------
int main() {
#if PROFILING
    // Define sizes
    const size_t heapSize = 1 << 18; // 262144
    const size_t groupSize = heapSize / 128; // 32768

    // Generate random data for d_vctCandHeap
    thrust::device_vector<Cand> d_vctCandHeap(heapSize);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> disInt(1, 10000);
    std::uniform_real_distribution<> disFloat(0.0f, 10000.0f);

    for (size_t i = 0; i < heapSize; ++i)
    {
        Cand temp = { disInt(gen), disInt(gen), disInt(gen), disFloat(gen) };
        d_vctCandHeap[i] = temp;
    }

    // Group begin indices
    thrust::device_vector<int> d_vctGroupBeginIndeces(groupSize);
    for (size_t i = 0; i < groupSize; ++i) 
    {
        d_vctGroupBeginIndeces[i] = i * (heapSize / groupSize);
    }

    // Permuted indices
    std::vector<int> indices(heapSize);
    std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, ..., heapSize - 1
    std::shuffle(indices.begin(), indices.end(), gen);

    thrust::device_vector<int> d_vctIndeces = indices;

    // Print permuted indices
    std::cout << "Permuted Indices: ";
    for (int i = 0; i < d_vctIndeces.size(); ++i)
    {
      //  std::cout << d_vctIndices[i] << " ";
    }
    std::cout << std::endl;

    thrust::device_vector<int> d_vctNumDelegates(groupSize);
    thrust::device_vector<int> d_vctNumDelegates0(groupSize);

    // Profile the function
    auto start = std::chrono::high_resolution_clock::now();
    select_delegates(d_vctCandHeap, d_vctIndeces, d_vctGroupBeginIndeces, d_vctNumDelegates);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Standard: find_num_delegates execution time: " << duration.count() << " microseconds" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    int threads_per_block = 256;
    int blocks_per_grid = d_vctGroupBeginIndeces.size();
     select_delegates_kernel << < blocks_per_grid, threads_per_block, threads_per_block* (sizeof(int) + sizeof(float)) >> >
         (thrust::raw_pointer_cast(d_vctCandHeap.data()),
             thrust::raw_pointer_cast(d_vctIndeces.data()),
             d_vctCandHeap.size(),
             thrust::raw_pointer_cast(d_vctGroupBeginIndeces.data()),
             thrust::raw_pointer_cast(d_vctNumDelegates0.data()));

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Custom kernel:find_num_delegates execution time: " << duration.count() << " microseconds" << std::endl;
    int idelta = 0;
    for (int i = 0; i < d_vctNumDelegates0.size(); ++i)
    {
        int  ideltat = fabs(d_vctNumDelegates0[i] - d_vctNumDelegates[i]);
        if (ideltat > idelta)
        {            
            idelta = ideltat;
            int iii = 0;
        }
    }
    d_vctNumDelegates = d_vctNumDelegates0;
    // Sort d_vctNumDelegates based on 'mt' member of d_vctCandHeap
    int member_offset = offsetof(Cand, mt);
    thrust::sort(d_vctNumDelegates.begin(), d_vctNumDelegates.end(),
        CompareCandMember(thrust::raw_pointer_cast(d_vctCandHeap.data()), member_offset));

    write_log(d_vctCandHeap, d_vctGroupBeginIndeces, d_vctIndeces, d_vctNumDelegates, "output.log");
#else
    // Sample data for d_vctCandHeap with 15 elements
    int QUantCand = 15;
    int QUantBeginIndeces = 3;
    thrust::device_vector<Cand> d_vctCandHeap(QUantCand);
    d_vctCandHeap[0] = { 1, 2, 3, 1.0f };
    d_vctCandHeap[1] = { 2, 3, 4, 2.0f };
    d_vctCandHeap[2] = { 3, 4, 5, 3.0f };
    d_vctCandHeap[3] = { 4, 5, 6, 0.5f };
    d_vctCandHeap[4] = { 5, 6, 7, 2.5f };
    d_vctCandHeap[5] = { 6, 7, 8, 1.5f };
    d_vctCandHeap[6] = { 7, 8, 9, 4.0f };
    d_vctCandHeap[7] = { 8, 9, 10, 3.5f };
    d_vctCandHeap[8] = { 9, 10, 11, 0.7f };
    d_vctCandHeap[9] = { 10, 11, 12, 1.3f };
    d_vctCandHeap[10] = { 11, 12, 13, 5.0f };
    d_vctCandHeap[11] = { 12, 13, 14, 3.3f };
    d_vctCandHeap[12] = { 13, 14, 15, 2.7f };
    d_vctCandHeap[13] = { 14, 15, 16, 4.5f };
    d_vctCandHeap[14] = { 15, 16, 17, 3.2f };

    // Group begin indices
    thrust::device_vector<int> d_vctGroupBeginIndeces = { 0, 3, 8 };

    // Permuted indices
    std::vector<int> indices = { 4, 6, 12, 7, 0, 2, 1, 8, 5, 14, 11, 9, 10, 3, 13 };// (QUantCand);
   // std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, ..., 14
   // std::random_device rd;
   // std::mt19937 g(rd());
   // std::shuffle(indices.begin(), indices.end(), g);

    thrust::device_vector<int> d_vctIndeces = indices;

    // Print permuted indices
    std::cout << "Permuted Indices: ";
    for (int i = 0; i < d_vctIndeces.size(); ++i) 
    {
        std::cout << d_vctIndeces[i] << " ";
    }
    std::cout << std::endl; 

    //--
    // Copy data back to host for printing
    thrust::host_vector<Cand> h_vctCandHeap(QUantCand);
    h_vctCandHeap = d_vctCandHeap;

    thrust::host_vector<int> h_vctIndeces(QUantCand);
    h_vctIndeces = d_vctIndeces;

    thrust::host_vector<int> h_vctGroupBeginIndeces(QUantBeginIndeces);
    h_vctGroupBeginIndeces = d_vctGroupBeginIndeces;

    

    std::cout << "d_arrCand:" << std::endl;
    for (int i = 0; i < QUantCand; ++i) {
        std::cout << i << " {" << h_vctCandHeap[i].mt << ", " << h_vctCandHeap[i].mdt << ", " << h_vctCandHeap[i].mwidth << ", " << h_vctCandHeap[i].msnr << "}" << std::endl;
    }

    std::cout << "d_arrIndeces: ";
    for (int i = 0; i < QUantCand; ++i) {
        std::cout << h_vctIndeces[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "d_arrGroupBeginIndecies: ";
    for (int i = 0; i < QUantBeginIndeces; ++i) {
        std::cout << h_vctGroupBeginIndeces[i] << " ";
    }
    std::cout  << std::endl;
    std::cout << "**************************" << std::endl;
    //--

    thrust::device_vector<int> d_vctNumDelegates(d_vctGroupBeginIndeces.size());
   select_delegates(d_vctCandHeap, d_vctIndeces, d_vctGroupBeginIndeces, d_vctNumDelegates);
    

    int threads_per_block = 4;
    int blocks_per_grid = d_vctGroupBeginIndeces.size();
    select_delegates_kernel << < blocks_per_grid, threads_per_block, threads_per_block* (sizeof(int) + sizeof(float)) >> >
        (thrust::raw_pointer_cast(d_vctCandHeap.data()),
            thrust::raw_pointer_cast(d_vctIndeces.data()),
            d_vctCandHeap.size(),
            thrust::raw_pointer_cast(d_vctGroupBeginIndeces.data()),
            thrust::raw_pointer_cast(d_vctNumDelegates.data()));

     // Write to log file
     write_log(d_vctCandHeap,d_vctGroupBeginIndeces, d_vctIndeces, d_vctNumDelegates, "output.log");


     // Print the same data to the console
     print_log(d_vctCandHeap, d_vctGroupBeginIndeces, d_vctIndeces, d_vctNumDelegates);
#endif

  
    return 0;
}



