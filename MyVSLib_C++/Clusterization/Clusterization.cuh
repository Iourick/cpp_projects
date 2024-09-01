#pragma once
#include "Candidate.cuh"
#include "Constants.h"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/host_vector.h>

namespace clusterization
{
    bool clusterization_main(const float* d_digitized_fdmt
        , const int Rows
        , const int Cols
        , const float& d_VAlTresh
        , const int WndWidth
        , const int* d_pbin_metrics
        , const std::string& filename
        , const float delt_bin
        , const float t_chunkBegin
        , bool bwrite = true
    );

    void gather_candidates_in_dynamicalArray(const float* d_arr_fdmt, const int  Rows, const int  Cols
        , const float& val_thresh, const int QUantPower2_Wnd, Cand** d_arrCand, unsigned int* h_pquantCand);

    void complete_clusterization(const Cand* d_parrCand
        , unsigned int quant_candidates
        , const int* d_pbin_metrics
        , const std::string& filename
        , const float delt_bin
        , const float t_chunkBegin
        , bool bwrite
    );

    __global__
        void digitize_kernel(fdmt_type_* d_arrfdmt, const int LEnarr);

    __global__
        void gather_candidates_in_fixedArray_kernel_v0(const float* d_arr_fdmt, const int  Cols, const float& val_thresh
            , const int QUantPower2_Wnd, unsigned  int* d_pimax_candidates_per_chunk, Cand* d_arrCand, unsigned int& d_quantCand);

    void write_log_v0(const Cand* d_arrCand
        , const int QUantCand
        , const std::string& filename
        , const float delt_bin
        , const float t_chunkBegin);

    void writeCandDeviceArrayToCSV(const Cand* d_arrCand, int h_quantCand, const std::string& filename);

    void writeCandHostArrayToCSV(const Cand* h_arrCand, int h_quantCand, const std::string& filename);

    void readFreddaCandFile(const std::string& filename, Cand* arrCand, size_t numLines);

    int  countCandidatesInFreddaFile(const std::string& filename);

    //--------------------------------------------------------------------------------
    bool gather_candidates_in_heap_with_plan(const float* d_fdmt
        , const int Rows
        , const int Cols
        , const float& d_VAlTresh
        , const int WndWidth
        , Cand** d_pparrCand// = nullptr
        , unsigned int* h_piQuantCand
    );

    __global__
        void gather_candidates_in_dynamicalArray_kernel(const float* d_arr_fdmt, const int  Cols, const float& val_thresh
            , const int QUantPower2_Wnd, unsigned int* d_pmaxquantCand, Cand* d_arrCand, unsigned int* d_pquantCand);
}



__global__
void calc_candidates(const float* d_arr_fdmt, const int  Cols, const float& val_thresh
    , const int QUantPower2_Wnd, unsigned int* d_pquantCand);



void printDeviceArray(const int* d_array, int length, const char* arrayName);

void fnc_grouping(const thrust::device_vector<Cand>& d_vctCandHeap
    , const int member_offset
    , const int& d_bin_metrics
    , thrust::device_vector<int>& d_vctIndices
    , thrust::device_vector<int>& h_vctGroupBeginIndices);

__global__
void calc_plan_and_values_for_regrouping_kernel(const Cand* d_arrCand
    , const  int* d_arrIndeces
    , const int QUantCand
    , const  int* d_arrGroupBeginIndecies
    , const int member_offset
    , const int& d_bin_metrics
    , int* d_arrValues
    , int* d_arrRegroupingPlan);

__global__
void regrouping_kernel(int* d_vctValues
    , const  int LEn_vctValues
    , const int* d_vctGroupBeginIndices
    , const int* d_vctRegroupingPlan
    , const  int LEn_vctGroupBeginIndices
    , const  int& d_bin_metrics
    , int* d_vctGroupBeginIndicesUpdated);

void print_parameters_for_fnc_grouping(const thrust::device_vector<Cand>& d_vctCandHeap,
    const int member_offset,
    const int& d_bin_metrics,
    const thrust::device_vector<int>& d_vctIndices,
    const thrust::device_vector<int>& d_vctGroupBeginIndices);

void print_delegates(const thrust::device_vector<Cand>& d_vctCandHeap
    , const thrust::device_vector<int>& d_vctNumDelegates, const char* arrayName);

__global__
void gather_candidates_in_heap_kernel(const float* d_arr_fdmt, const int  Cols, const float& val_thesh
    , const int WndWidth, const unsigned int* d_arr_plan, Cand* d_arrCand_Heap);

void print_input_clusterization(const float* d_digitized_fdmt, const int Rows, const int Cols,
    const float& VAlTresh, const int WndWidth,
    const int* d_pbin_metrics, const std::string& filename);

__global__
void do_plan_kernel(const float* d_arr, const int  Cols, const float& val_thresh
    , const int WndWidth, unsigned int* d_arr_plan);

void checkCudaError1(const char* msg);

void print_regrouping_input(int* d_vctValues,
    const int LEn_vctValues,
    const int* d_vctGroupBeginIndices,
    const int* d_vctRegroupingPlan,
    const int LEn_vctGroupBeginIndices);

void print_candidates_less_threshold(const thrust::device_vector<Cand>& d_vctCandHeap, const float* d_VAlTresh);

void fnc_grouping_(const Cand* d_parrCand
    , const int QUantCandidates
    , const int member_offset
    , const int& d_bin_metrics
    , thrust::device_vector<int>& d_vctIndeces
    , thrust::device_vector<int>& d_vctOffset);

void fnc_grouping_(const Cand* d_parrCand
    , const int QUantCandidates
    , const int member_offset
    , const int& d_bin_metrics
    , thrust::device_vector<int>& d_vctIndeces
    , thrust::device_vector<int>& d_vctOffset);

int apb(int a, int b);

