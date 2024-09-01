#ifndef SEGMENTED_SORT_CUH
#define SEGMENTED_SORT_CUH
#include "Candidate.cuh"
//struct SCand {
//    int mt;
//    int mdt;
//    int mwidth;
//    float msnr;
//};
//using Cand = SCand;

namespace segmented_sort
{
    void sort_subarrays(const thrust::device_vector<Cand>& d_vctCandHeap,
        thrust::device_vector<int>& d_vctGroupBeginIndices,
        thrust::device_vector<int>& d_indices,
        int member_offset);

    void sort_subarrays_cub(const thrust::device_vector<Cand>& d_vctCandHeap,
        thrust::device_vector<int>& d_vctOffset,
        thrust::device_vector<int>& d_indices,
        int member_offset);

    void sort_subarrays_cub_(const Cand* d_parrCand,
        const int QUantElements,
        thrust::device_vector<int>& d_vctOffset,
        thrust::device_vector<int>& d_indices,
        int member_offset);
};

//struct CompareCandMember {
//    const Cand* d_vctCandHeap;
//    int member_offset;
//
//    CompareCandMember(const Cand* _d_vctCandHeap, int _member_offset)
//        : d_vctCandHeap(_d_vctCandHeap), member_offset(_member_offset) {}
//
//    __host__ __device__
//        bool operator()(const int& idx1, const int& idx2) const {
//        const char* base1 = reinterpret_cast<const char*>(&d_vctCandHeap[idx1]);
//        const char* base2 = reinterpret_cast<const char*>(&d_vctCandHeap[idx2]);
//        int value1 = *reinterpret_cast<const int*>(base1 + member_offset);
//        int value2 = *reinterpret_cast<const int*>(base2 + member_offset);
//        return value1 < value2;
//    }
//};

__global__ void validate_access(int* indices, Cand* d_vctCandHeap, int member_offset, int size);




void checkCudaErrors(cudaError_t err);

__global__
void extract_keys_kernel(const Cand* d_parrCand
    , const int QUantElements
    , const int* d_parrIndeces
    , int member_offset
    , int* d_arrKeys);

#endif // SEGMENTED_SORT_CUH