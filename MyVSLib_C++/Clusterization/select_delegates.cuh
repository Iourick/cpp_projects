#pragma once
#include "Candidate.cuh"
#include <thrust/device_vector.h>
namespace select_delegates
{
    __global__
        void select_delegates_kernel(const Cand* d_arrCandHeap,
            int* d_arrIndices,
            const int QUantCand,
            int* d_arrGroupBeginIndeces,
            int* d_arrNumDelegates);

    void select_delegates(const thrust::device_vector<Cand>& d_vctCandHeap,
        const thrust::device_vector<int>& d_vctIndices,
        const thrust::device_vector<int>& d_vctGroupBeginIndices,
        thrust::device_vector<int>& d_vctNumDelegates);

    void write_log(const Cand* d_parrCand,
        const int QUantCandidates,
        const thrust::device_vector<int>& d_vctGroupBeginIndices,
        const thrust::device_vector<int>& d_vctIndices,
        const thrust::device_vector<int>& d_vctNumDelegates,
        const std::string& filename
        , const float delt_bin
        , const float t_chunkBegin);
};

void print_log(const thrust::device_vector<Cand>& d_vctCandHeap,
    const thrust::device_vector<int>& d_vctGroupBeginIndices,
    const thrust::device_vector<int>& d_vctIndices,
    const thrust::device_vector<int>& d_vctNumDelegates);

