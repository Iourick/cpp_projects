#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <iostream>
#include <fstream>
#include <iomanip> // Include this header for std::setw
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include "Clusterization.cuh"
#include "select_delegates.cuh"
#include "Segmented_sort.cuh"
#include <sstream>

 

bool clusterization::clusterization_main(const float* d_fdmt
    , const int Rows
    , const int Cols
    , const float& d_VAlTresh
    , const int WndWidth
    , const int* d_pbin_metrics
    , const std::string& filename
    , const float delt_bin
    , const float t_chunkBegin
    , bool bwrite
)
{
    unsigned int quant_candidates = 0;
    Cand* d_parrCand = nullptr;
    gather_candidates_in_heap_with_plan(d_fdmt
        , Rows
        , Cols
        , d_VAlTresh
        , WndWidth
        , &d_parrCand// = nullptr
        , &quant_candidates
    );
    if (0 == quant_candidates)
    {
        return false;
    }
    complete_clusterization(d_parrCand
        , quant_candidates
        , d_pbin_metrics
        , filename
        , delt_bin
        , t_chunkBegin
        , bwrite
    );
    // !3

    cudaFree(d_parrCand);
}
//------------------------------------------------------------------------------------------------
void clusterization::gather_candidates_in_dynamicalArray(const float* d_arr_fdmt, const int  Rows, const int  Cols
    , const float& val_thresh, const int QUantPower2_Wnd, Cand** d_pparrCand, unsigned int* h_pquantCand)
{
    // 1. memory preparations
    unsigned int it = 0;
    unsigned int* d_pquantCand = nullptr;
    cudaMalloc((void**)&d_pquantCand, sizeof(unsigned int));
    cudaMemcpy(d_pquantCand, &it, sizeof(unsigned int), cudaMemcpyHostToDevice);
    // !1

    // 2. calculations of quantity of candidates
    const dim3 blockSize(256, 1, 1);
    const dim3 gridSize((Cols + blockSize.x - 1) / blockSize.x, Rows, 1);
    calc_candidates << < gridSize, blockSize >> > (d_arr_fdmt, Cols, val_thresh
        , QUantPower2_Wnd, d_pquantCand);
    // !2

    // 3. transfere d_pquantCand to host
    cudaMemcpy(h_pquantCand, d_pquantCand, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMalloc((void**)d_pparrCand, (*h_pquantCand) * sizeof(Cand));
    // !3

    // 4.allocation memory for temporary variable - current candidates quantity
    unsigned int* d_pCurquantCand = nullptr;
    cudaMalloc((void**)&d_pCurquantCand, sizeof(unsigned int));
    cudaMemcpy(d_pCurquantCand, &it, sizeof(unsigned int), cudaMemcpyHostToDevice);
    gather_candidates_in_dynamicalArray_kernel << < gridSize, blockSize >> >
        (d_arr_fdmt, Cols, val_thresh, QUantPower2_Wnd, d_pquantCand, *d_pparrCand
            , d_pCurquantCand);
    // !4
    cudaFree(d_pCurquantCand);
    cudaFree(d_pquantCand);

}

//-----------------------------------------------------------------------------
void clusterization::complete_clusterization(const Cand* d_parrCand
    , unsigned int quant_candidates
    , const int* d_pbin_metrics
    , const std::string& filename
    , const float delt_bin
    , const float t_chunkBegin
    , bool bwrite
)
{
    // 4. Create a thrust::device_vector of indices. {0,1,2..,d_vctCandHeap.size()-1}
    thrust::device_vector<int> d_vctIndeces(quant_candidates);
    thrust::sequence(d_vctIndeces.begin(), d_vctIndeces.end());
    // !4

    // 5. initializing for grouping process.
    // In vector d_vctGroupBeginIndeces we store indeces of first element of each group
    // we don't sort elements of the vector d_vctCandHeap, but sort elements of vector  d_vctIndeces
    int quantityGroups = 1;
    thrust::device_vector<int> d_vctOffset(quantityGroups, 0);
    d_vctOffset.push_back(d_vctIndeces.size());
    // !5   


    // 6. mt-axis grouping 
    int member_offset = offsetof(Cand, mt);
    fnc_grouping_(d_parrCand, quant_candidates, member_offset, d_pbin_metrics[0], d_vctIndeces, d_vctOffset);

    // !6

    // 7. mdt-axis grouping 
    member_offset = offsetof(Cand, mdt);
    fnc_grouping_(d_parrCand, quant_candidates, member_offset, d_pbin_metrics[1], d_vctIndeces, d_vctOffset);
    // !7    

    // 8. mwidth-axis grouping 
    member_offset = offsetof(Cand, mwidth);
    fnc_grouping_(d_parrCand, quant_candidates, member_offset, d_pbin_metrics[2], d_vctIndeces, d_vctOffset);
    // !8   

    // 9. calculation delegates - pick up member of each group with maximal value of .msnr 
    thrust::device_vector<int>  d_vctNumDelegates(d_vctOffset.size() - 1);

    int threads_per_block = 64;
    int blocks_per_grid = d_vctOffset.size() - 1;
    select_delegates::select_delegates_kernel << < blocks_per_grid, threads_per_block, threads_per_block* (sizeof(int) + sizeof(float)) >> >
        (d_parrCand,
            thrust::raw_pointer_cast(d_vctIndeces.data()),
            quant_candidates,
            thrust::raw_pointer_cast(d_vctOffset.data()),
            thrust::raw_pointer_cast(d_vctNumDelegates.data()));


    // !9

    //10. Sort d_vctNumDelegates based on 'mt' member of d_vctCandHeap
    member_offset = offsetof(Cand, mt);
    thrust::sort(d_vctNumDelegates.begin(), d_vctNumDelegates.end(),
        CompareCandMember(d_parrCand, member_offset));

    // !10

    if (bwrite)
    {
        // 11.write Delegates in the file "filename"
        select_delegates::write_log(d_parrCand,
            quant_candidates,
            d_vctOffset,
            d_vctIndeces,
            d_vctNumDelegates,
            filename
            , delt_bin
            , t_chunkBegin);

    }
}
//-------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------
bool clusterization::gather_candidates_in_heap_with_plan(const float* d_fdmt
    , const int Rows
    , const int Cols
    , const float& d_VAlTresh
    , const int WndWidth
    , Cand** d_pparrCand// = nullptr
    , unsigned int* h_piQuantCand
)
{

    // 1. calculate "plan" - how much Candidates produces each element of fdmt matrix 
    thrust::device_vector<unsigned int> d_plan(Rows * Cols);
    const dim3 blockSize(256, 1, 1);
    const dim3 gridSize((Cols + blockSize.x - 1) / blockSize.x, Rows, 1);
    do_plan_kernel << < gridSize, blockSize >> > (d_fdmt, Cols, d_VAlTresh
        , WndWidth, thrust::raw_pointer_cast(d_plan.data()));
    // !1

    // 2. calculate cummulative sum of plan
    thrust::inclusive_scan(d_plan.begin(), d_plan.end(), d_plan.begin());
    // !2

    // 3.  Create heap of candidates       
    *h_piQuantCand = d_plan.back();
    if (0 == (*h_piQuantCand))
    {
        return false;
    }
    thrust::device_vector<Cand> d_vctCandHeap(*h_piQuantCand);
    cudaMalloc((void**)d_pparrCand, (*h_piQuantCand) * sizeof(Cand));
    gather_candidates_in_heap_kernel << < gridSize, blockSize >> > (d_fdmt, Cols, d_VAlTresh
        , WndWidth, thrust::raw_pointer_cast(d_plan.data()), *d_pparrCand);
    return true;
}

//--------------------------------------------------------------------------

__global__
    void clusterization::digitize_kernel(fdmt_type_* d_arrfdmt, const int LEnarr)
{

    for (int i = threadIdx.x; i < LEnarr; i += blockDim.x)
    {
        d_arrfdmt[i] = floorf(d_arrfdmt[i]);
    }
}

//--------------------------------------------------------------------------------------------
__global__
    void clusterization::gather_candidates_in_fixedArray_kernel_v0(const float* d_arr_fdmt, const int  Cols
        , const float& val_thresh, const int QUantPower2_Wnd, unsigned  int* d_pimax_candidates_per_chunk
        , Cand* d_arrCand, unsigned int& d_quantCand)

{
    const int iCol = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCol >= Cols)
    {
        return;
    }
    int iRow = blockIdx.y;
    int numElem = iRow * Cols + iCol;

    //------------------------------------
    int inum = 0;
    const int WndWidth = 1 << QUantPower2_Wnd;
    int ipowCur = 0;
    int iLenWindCur = 1;
    float sig2 = 0;
    for (int iw = 0; iw < WndWidth; ++iw)
    {
        if (iCol + iw < Cols)
        {
            sig2 += d_arr_fdmt[numElem + iw];
            if ((iw + 1) == iLenWindCur)
            {
                float temp = fdividef(sig2, sqrtf(static_cast<float>(iw + 1)));
                if (temp > val_thresh)
                {
                    unsigned int inum = atomicInc(&d_quantCand, *d_pimax_candidates_per_chunk);
                    d_arrCand[inum].mt = iCol;
                    d_arrCand[inum].mdt = iRow;
                    d_arrCand[inum].mwidth = ipowCur;
                    d_arrCand[inum].msnr = temp;

                }
                ++ipowCur;
                iLenWindCur *= 2;
            }

        }
        else
        {
            break;
        }
    }
}
//d_quantCand - should be allocated on GPU

//----------------------------------------------------------------------------------------------
void clusterization::write_log_v0(const Cand* d_arrCand
    , const int QUantCand
    , const std::string& filename
    , const float delt_bin
    , const float t_chunkBegin)
{

    // Copy device vectors to host vectors
    Cand* h_arrCand = (Cand*)malloc(QUantCand * sizeof(Cand));
    cudaMemcpy(h_arrCand, d_arrCand, QUantCand * sizeof(Cand), cudaMemcpyDeviceToHost);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    // Open log file
    std::ofstream logFile(filename);

    // Write header
    logFile << std::setw(10) << "N"
        << std::setw(10) << "t, bin"
        << std::setw(10) << "t, sec"
        << std::setw(10) << "dedisp"
        << std::setw(10) << "width"
        << std::setw(10) << "SNR"
        << std::endl;

    // Write data
    for (int i = 0; i < QUantCand; ++i)
    {
        logFile << std::setw(10) << i + 1
            << std::setw(10) << h_arrCand[i].mt
            << std::setw(10) << t_chunkBegin + h_arrCand[i].mt * delt_bin
            << std::setw(10) << h_arrCand[i].mdt
            << std::setw(10) << h_arrCand[i].mwidth
            << std::setw(10) << h_arrCand[i].msnr
            << std::endl;
    }
    logFile.close();
    free(h_arrCand);
};
//------------------------------------------------------
void clusterization::writeCandDeviceArrayToCSV(const Cand* d_arrCand, int h_quantCand, const std::string& filename)
{
    // Allocate host array for copying back
    Cand* h_arrCand = new Cand[h_quantCand];

    // Copy device array to host array
    cudaMemcpy(h_arrCand, d_arrCand, h_quantCand * sizeof(Cand), cudaMemcpyDeviceToHost);
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    // Write header
    file << "mt,mdt,mwidth,msnr\n";

    // Write each element in a line
    for (size_t i = 0; i < h_quantCand; ++i) {
        file << h_arrCand[i].mt << ','
            << h_arrCand[i].mdt << ','
            << h_arrCand[i].mwidth << ','
            << h_arrCand[i].msnr << '\n';
    }

    file.close();
    delete[]h_arrCand;
}
//----------------------------------------------------------------------
void clusterization::writeCandHostArrayToCSV(const Cand* h_arrCand, int h_quantCand, const std::string& filename)
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    // Write header
    file << "mt,mdt,mwidth,msnr\n";

    // Write each element in a line
    for (size_t i = 0; i < h_quantCand; ++i) {
        file << h_arrCand[i].mt << ','
            << h_arrCand[i].mdt << ','
            << h_arrCand[i].mwidth << ','
            << h_arrCand[i].msnr << '\n';
    }

    file.close();
}


//---------------------------------------------------------
int clusterization::countCandidatesInFreddaFile(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return 0;
    }

    std::string line;
    int count = 0;

    // Skip the header line
    std::getline(file, line);

    // Count the remaining lines
    while (std::getline(file, line)) {
        ++count;
    }

    file.close();
    return count;
}
//------------------------------------------
void clusterization::readFreddaCandFile(const std::string& filename, Cand* arrCand, size_t numLines)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    std::string line;
    // Skip the first line (header)
    std::getline(file, line);

    // Read each line and fill the array
    size_t index = 0;
    while (std::getline(file, line) && index < numLines) {
        std::istringstream iss(line);
        float snr;
        int sampno, boxcar, idt;
        float dm, secs;
        int beamno;
        if (iss >> snr >> sampno >> secs >> boxcar >> idt >> dm >> beamno) {
            arrCand[index].mt = sampno;
            arrCand[index].mdt = idt;
            arrCand[index].mwidth = boxcar;
            arrCand[index].msnr = snr;
            ++index;
        }
    }
    file.close();
}


//---------------------------------------------------------------------------------
__global__
    void clusterization::gather_candidates_in_dynamicalArray_kernel(const float* d_arr_fdmt, const int  Cols, const float& val_thresh
        , const int QUantPower2_Wnd, unsigned int* d_pmaxquantCand, Cand* d_arrCand, unsigned int* d_pquantCand)
{
    const int iCol = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCol >= Cols)
    {
        return;
    }
    int iRow = blockIdx.y;
    int numElem = iRow * Cols + iCol;

    //------------------------------------    
    const int WndWidth = 1 << QUantPower2_Wnd;
    int ipowCur = 0;
    int iLenWindCur = 1;
    float sig2 = 0;
    for (int iw = 0; iw < WndWidth; ++iw)
    {
        if (iCol + iw < Cols)
        {
            sig2 += d_arr_fdmt[numElem + iw];
            if ((iw + 1) == iLenWindCur)
            {
                float temp = fdividef(sig2, sqrtf(static_cast<float>(iw + 1)));
                if (temp > val_thresh)
                {
                    // printf("*d_pmaxquantCand = %i\n", d_pmaxquantCand[0]);
                    unsigned int inum = atomicInc(d_pquantCand, *d_pmaxquantCand);
                    d_arrCand[inum].mt = iCol;
                    d_arrCand[inum].mdt = iRow;
                    d_arrCand[inum].mwidth = ipowCur;
                    d_arrCand[inum].msnr = temp;
                }
                ++ipowCur;
                iLenWindCur *= 2;
            }
        }
        else
        {
            break;
        }
    }
}

//---------------------------------------------------------------------------------------------
__global__
void calc_candidates(const float* d_arr_fdmt, const int  Cols, const float& val_thresh
    , const int QUantPower2_Wnd, unsigned int* d_pquantCand)
{
    const int iCol = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCol >= Cols)
    {
        return;
    }
    int iRow = blockIdx.y;
    int numElem = iRow * Cols + iCol;
    //------------------------------------
    int imax_candidates_per_chunk = 1 << 31;
    const int WndWidth = 1 << QUantPower2_Wnd;
    int ipowCur = 0;
    int iLenWindCur = 1;
    float sig2 = 0;
    for (int iw = 0; iw < WndWidth; ++iw)
    {
        if (iCol + iw < Cols)
        {
            sig2 += d_arr_fdmt[numElem + iw];
            if ((iw + 1) == iLenWindCur)
            {
                float temp = fdividef(sig2, sqrtf(static_cast<float>(iw + 1)));
                if (temp > val_thresh)
                {
                    unsigned int inum = atomicInc(d_pquantCand, imax_candidates_per_chunk);
                }
                ++ipowCur;
                iLenWindCur *= 2;
            }
        }
        else
        {
            break;
        }
    }
}

void printDeviceArray(const int* d_array, int length, const char* arrayName)
{
    // Allocate host array
    int* h_array = new int[length];

    // Copy device array to host
    cudaMemcpy(h_array, d_array, length * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the array
    std::cout << arrayName << ": " << std::endl;
    for (int i = 0; i < length; ++i) {
        std::cout << h_array[i] << " ";
        if ((i + 1) % 10 == 0) {
            std::cout << std::endl;  // Newline after every 10 values
        }
    }
    if (length % 10 != 0) {
        std::cout << std::endl;  // Newline if the last row isn't complete
    }

    // Free host array
    delete[] h_array;
}
//---------------------------------------------------------------------------------------------------------------------------------------
 void print_candidates_less_threshold(const thrust::device_vector<Cand>& d_vctCandHeap, const float *d_VAlTresh)
 {
        // Copy the threshold value from device to host
    float h_VAlTresh;
    cudaMemcpy(&h_VAlTresh, d_VAlTresh, sizeof(float), cudaMemcpyDeviceToHost);

    // Copy the device vector to a host vector
    thrust::host_vector<Cand> h_vctCandHeap = d_vctCandHeap;

    // Print the candidates
    std::cout << "Candidates with msnr less than " << h_VAlTresh << ":\n";
    for (size_t i = 0; i < h_vctCandHeap.size(); ++i)
    {
        if (h_vctCandHeap[i].msnr < h_VAlTresh)
        {
            std::cout << "Candidate " << i << ": mt: " << h_vctCandHeap[i].mt
                << ", mdt: " << h_vctCandHeap[i].mdt
                << ", mwidth: " << h_vctCandHeap[i].mwidth
                << ", msnr: " << h_vctCandHeap[i].msnr << "\n";
        }
    }
}

//-----------------------------------------------------------------------------------------------------------------------------------
void print_delegates(const thrust::device_vector<Cand>& d_vctCandHeap, const thrust::device_vector<int>& d_vctNumDelegates, const char* arrayName) 
{
    // Copy the device vectors to host vectors
    thrust::host_vector<Cand> h_vctCandHeap = d_vctCandHeap;
    thrust::host_vector<int> h_vctNumDelegates = d_vctNumDelegates;

    // Print the header
    std::cout << "Array Name: " << arrayName << "\n";

    // Print the structures
    for (size_t i = 0; i < h_vctNumDelegates.size(); ++i) 
    {
        int index = h_vctNumDelegates[i];
        if (index < h_vctCandHeap.size())
        {
            Cand c = h_vctCandHeap[index];
            std::cout << "Delegate " << i << ":\n";
            std::cout << "mt: " << c.mt << ", mdt: " << c.mdt << ", mwidth: " << c.mwidth << ", msnr: " << c.msnr << "\n";
        }
        else 
        {
            std::cout << "Index out of bounds: " << index << "\n";
        }
    }
}
/*----------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------*/
void checkCudaError1(const char* msg)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
    {
        std::cerr << "CUDA Error: " << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}
////----------------------------------------------------
__global__
void do_plan_kernel(const float* d_arr, const int  Cols, const float &val_thresh
    , const int WndWidth, unsigned int* d_arr_plan)
{
    const int icol = blockIdx.x * blockDim.x + threadIdx.x;
    if (icol >= Cols)
    {
        return;
    }
    int nrow = blockIdx.y;
    int numelem = nrow * Cols + icol;   
    int inum = 0;
    //------------------------------------
    for (int iw = 1; iw < WndWidth + 1; ++iw)
    {
        if (icol + iw - 1 < Cols)
        {
            float sig2 = 0;

            for (int j = 0; j < iw; ++j)
            {
                sig2 += d_arr[numelem + j];

            }
            float temp = fdividef(sig2, sqrtf(static_cast<float>(iw)));
            if (temp >  val_thresh )
            {                
                inum++;
            }
        }
    }
    d_arr_plan[numelem] = inum;
}

////----------------------------------------------------
__global__
void gather_candidates_in_heap_kernel(const float* d_arr_fdmt, const int  Cols, const float& val_thresh
    , const int WndWidth, const unsigned int* d_arr_plan, Cand* d_arrCand_Heap)
{
    const int iCol = blockIdx.x * blockDim.x + threadIdx.x;
    if (iCol >= Cols)
    {
        return;
    }
    int iRow = blockIdx.y;
    int numElem = iRow * Cols + iCol;
    int quantCandidates = (numElem == 0) ? d_arr_plan[0] : d_arr_plan[numElem] - d_arr_plan[numElem - 1];
    if (0 == quantCandidates)
    {
        return;
    }
    Cand* pcand = (numElem == 0) ? d_arrCand_Heap : d_arrCand_Heap + d_arr_plan[numElem - 1];
    //------------------------------------
    int inum = 0;
    bool bstop = false;
    for (int iw = 1; iw < WndWidth + 1; ++iw)
    {
        if (iCol + iw - 1 < Cols)
        {
            float sig2 = 0;

            for (int j = 0; j < iw; ++j)
            {
                sig2 += d_arr_fdmt[numElem + j];
            }
            float temp = fdividef(sig2, sqrtf(static_cast<float>(iw)));
            if (temp > val_thresh)
            {
                pcand[inum].mt = iCol;
                pcand[inum].mdt = iRow;
                pcand[inum].mwidth = iw;
                pcand[inum].msnr = temp;
                inum++; 
                if (inum == quantCandidates)
                {
                    break;
                }
            }
        }
    }
}

//----------------------------------------------------------------------------
//void sort_subarrays(const thrust::device_vector<Cand>& d_vctCandHeap,
//    thrust::device_vector<int>& d_vctGroupBeginIndices,
//    thrust::device_vector<int>& d_indices,
//    int member_offset)
//{
//    int* pind = thrust::raw_pointer_cast(d_indices.data());
//    thrust::host_vector<int> h_group_begin_indices = d_vctGroupBeginIndices;
//    for (int i = 0; i < h_group_begin_indices.size(); ++i)
//    {        
//        int start_index = h_group_begin_indices[i];     
//
//        int end_index = (i == h_group_begin_indices.size() - 1) ? d_indices.size() : h_group_begin_indices[i + 1];
//        int length = end_index - start_index;
//        thrust::device_ptr<int> dev_ptr_start = thrust::device_pointer_cast(pind + start_index);
//        thrust::device_ptr<int> dev_ptr_end = dev_ptr_start + length;
//
//        // Reintroduce the sorting step with additional checks
//        try {
//            thrust::sort(thrust::device, dev_ptr_start, dev_ptr_end,
//                CompareCandMember(thrust::raw_pointer_cast(d_vctCandHeap.data()), member_offset));
//
//        }
//        catch (std::runtime_error& e) {
//            std::cerr << "Runtime error during sorting: " << e.what() << std::endl;
//        }
//    }
//}
//--------------------------------------------------------------------------------------------------


void print_parameters_for_fnc_grouping(const thrust::device_vector<Cand>& d_vctCandHeap,
    const int member_offset,
    const int& d_bin_metrics,
    const thrust::device_vector<int>& d_vctIndeces,
    const thrust::device_vector<int>& d_vctOffset)
{
    // Copy device vectors to host vectors
    thrust::host_vector<Cand> h_vctCandHeap = d_vctCandHeap;
    thrust::host_vector<int> h_vctIndices = d_vctIndeces;
    thrust::host_vector<int> h_vctOffset = d_vctOffset;

    // Print d_vctCandHeap
    int in = 0;
    std::cout << "d_vctCandHeap: " << std::endl;
    for (const auto& cand : h_vctCandHeap) {
        std::cout << in << "   mt: " << cand.mt << ", mdt: " << cand.mdt
            << ", mwidth: " << cand.mwidth << ", msnr: " << cand.msnr << std::endl;
        ++in;
    }

    // Print member_offset
    std::cout << "member_offset: " << member_offset << std::endl;

    // Print d_bin_metrics
    int* h_bin_metrics = (int*)malloc(sizeof(int));
    cudaMemcpy(h_bin_metrics, &d_bin_metrics, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "d_bin_metrics: " << h_bin_metrics[0] << std::endl;
    free(h_bin_metrics);

    // Print d_vctIndeces
    std::cout << "d_vctIndeces: " << std::endl;
    for (const auto& index : h_vctIndices) {
        std::cout << index << "  ";
    }
    std::cout << std::endl;
    // Print d_vctGroupBeginIndices
    std::cout << "d_vctOffset: " << std::endl;
    for (const auto& index : h_vctOffset)
    {
        std::cout << index << std::endl;
    }
}

//--------------------------------------------------------------------
void fnc_grouping(const thrust::device_vector<Cand>& d_vctCandHeap
    , const int member_offset
    , const int& d_bin_metrics
    , thrust::device_vector<int>& d_vctIndeces
    , thrust::device_vector<int>& d_vctOffset)
{
    const int QUantCurrentGroups = d_vctOffset.size() - 1;
    segmented_sort::sort_subarrays_cub(d_vctCandHeap
        , d_vctOffset, d_vctIndeces, member_offset);

    thrust::device_vector<int> d_vctValues(d_vctIndeces.size());
    thrust::device_vector<int> d_vctRegroupingPlan(QUantCurrentGroups);
    int threads_per_block0 = 1024;
    calc_plan_and_values_for_regrouping_kernel << < QUantCurrentGroups, 1024, threads_per_block0 * sizeof(int) >> >
        (thrust::raw_pointer_cast(d_vctCandHeap.data())
            , thrust::raw_pointer_cast(d_vctIndeces.data())
            , d_vctIndeces.size()
            , thrust::raw_pointer_cast(d_vctOffset.data())
            , member_offset
            , d_bin_metrics
            , thrust::raw_pointer_cast(d_vctValues.data())
            , thrust::raw_pointer_cast(d_vctRegroupingPlan.data())
            );

    thrust::inclusive_scan(d_vctRegroupingPlan.begin(), d_vctRegroupingPlan.end(), d_vctRegroupingPlan.begin());

    //  int iii = d_vctRegroupingPlan.back();//!!

    thrust::device_vector<int> d_vctOffsetUpdated(d_vctRegroupingPlan.back() + 1);
    int threads_per_block = 1024;
    int blocks_per_grid = ((QUantCurrentGroups + 1) + threads_per_block - 1) / threads_per_block;
    regrouping_kernel << < blocks_per_grid, threads_per_block >> > (
        thrust::raw_pointer_cast(d_vctValues.data())
        , d_vctValues.size()
        , thrust::raw_pointer_cast(d_vctOffset.data())
        , thrust::raw_pointer_cast(d_vctRegroupingPlan.data())
        , QUantCurrentGroups
        , d_bin_metrics
        , thrust::raw_pointer_cast(d_vctOffsetUpdated.data())
        );

    d_vctOffset = d_vctOffsetUpdated;
}

//--------------------------------------------------------------------
void fnc_grouping_(const Cand* d_parrCand
    , const int QUantCandidates
    , const int member_offset
    , const int& d_bin_metrics
    , thrust::device_vector<int>& d_vctIndeces
    , thrust::device_vector<int>& d_vctOffset)
{
    const int QUantCurrentGroups = d_vctOffset.size() - 1;
    segmented_sort::sort_subarrays_cub_(d_parrCand, QUantCandidates
        , d_vctOffset, d_vctIndeces, member_offset);

    thrust::device_vector<int> d_vctValues(d_vctIndeces.size());
    thrust::device_vector<int> d_vctRegroupingPlan(QUantCurrentGroups);
    int threads_per_block0 = 1024;
    calc_plan_and_values_for_regrouping_kernel << < QUantCurrentGroups, 1024, threads_per_block0 * sizeof(int) >> >
        (     d_parrCand
            , thrust::raw_pointer_cast(d_vctIndeces.data())
            , d_vctIndeces.size()
            , thrust::raw_pointer_cast(d_vctOffset.data())
            , member_offset
            , d_bin_metrics
            , thrust::raw_pointer_cast(d_vctValues.data())
            , thrust::raw_pointer_cast(d_vctRegroupingPlan.data())
            );

    thrust::inclusive_scan(d_vctRegroupingPlan.begin(), d_vctRegroupingPlan.end(), d_vctRegroupingPlan.begin());

    //  int iii = d_vctRegroupingPlan.back();//!!

    thrust::device_vector<int> d_vctOffsetUpdated(d_vctRegroupingPlan.back() + 1);
    int threads_per_block = 1024;
    int blocks_per_grid = ((QUantCurrentGroups + 1) + threads_per_block - 1) / threads_per_block;
    regrouping_kernel << < blocks_per_grid, threads_per_block >> > (
        thrust::raw_pointer_cast(d_vctValues.data())
        , d_vctValues.size()
        , thrust::raw_pointer_cast(d_vctOffset.data())
        , thrust::raw_pointer_cast(d_vctRegroupingPlan.data())
        , QUantCurrentGroups
        , d_bin_metrics
        , thrust::raw_pointer_cast(d_vctOffsetUpdated.data())
        );

    d_vctOffset = d_vctOffsetUpdated;
}
//------------------------------------------------------------------------------------------------
// INPUT:
// 1.d_vctValues[LEn_vctValues] - this array consists of values
// 2.LEn_vctValues - size of the d_vctValues
// 3. d_vctGroupBeginIndices[LEn_vctGroupBeginIndices] - this array consists of numbers first elements for each group
//    suppose d_vctValues is splited for LEn_vctGroupBeginIndices groups
//    so, d_vctGroupBeginIndices[i] is number of first element of group number i
//  4. LEn_vctGroupBeginIndices - quantity of groups
// 5. d_vctRegroupingPlan [ LEn_vctGroupBeginIndices] - this array is cummulative sum computed 
//      of array consisting of quantity of subgroups for each group
// 6. d_bin_metrics - value to split array for subgroups.
//OUTPUT:
//d_vctGroupBeginIndicesUpdated - this array consists of numbers of beginning of each new group
// CALLING SAMPLE:
// int threads_per_block = 1024
// int blocks_per_grid = (LEn_vctGroupBeginIndices +  threads_per_block -1)/ threads_per_block ;
// <<< blocks_per_grid,  threads_per_block>>>
__global__
void regrouping_kernel(int* d_vctValues
    , const  int LEn_vctValues
    , const int* d_vctOffset
    , const int* d_vctRegroupingPlan
    , const  int QUantCurGroups
    , const  int& d_bin_metrics
    , int* d_vctOffsetUpdated)
{
    int numGroup = threadIdx.x + blockDim.x * blockIdx.x;
    if (numGroup > QUantCurGroups)
    {
        return;
    }
    if (numGroup == QUantCurGroups)
    {
        d_vctOffsetUpdated[d_vctRegroupingPlan[QUantCurGroups - 1]] = LEn_vctValues;
        return; 
    }

    int* pVal = &d_vctValues[d_vctOffset[numGroup]];
    int lengthGroup = d_vctOffset[numGroup + 1] - d_vctOffset[numGroup];
    int* pOffsetUpdated = (numGroup == 0) ? d_vctOffsetUpdated : &d_vctOffsetUpdated[d_vctRegroupingPlan[numGroup - 1]];

    pOffsetUpdated[0] = d_vctOffset[numGroup];

    int idx = 1;

    ++pVal;
    for (int i = 1; i < lengthGroup; ++i)
    {
        if (((*pVal) - (*(pVal - 1))) > d_bin_metrics)
        {
            pOffsetUpdated[idx] = pOffsetUpdated[0] + i;
            ++idx;
        }
        ++pVal;

    }
}

//__global__
//void regrouping_kernel(int* d_vctValues
//    , const  int LEn_vctValues
//    , const int* d_vctGroupBeginIndices
//    , const int* d_vctRegroupingPlan
//    , const  int LEn_vctGroupBeginIndices
//    , const  int& d_bin_metrics
//    , int* d_vctGroupBeginIndicesUpdated)
//{
//    int numGroup = threadIdx.x + blockDim.x * blockIdx.x;
//    if (numGroup >= LEn_vctGroupBeginIndices)
//    {
//        return;
//    }
//    int* pVal = &d_vctValues[d_vctGroupBeginIndices[numGroup]];
//    int lengthGroup = (numGroup == (LEn_vctGroupBeginIndices - 1)) ? LEn_vctValues - d_vctGroupBeginIndices[numGroup] : d_vctGroupBeginIndices[numGroup + 1] - d_vctGroupBeginIndices[numGroup];
//    int* pGroupBeginIndicesUpdated = (numGroup == 0) ? d_vctGroupBeginIndicesUpdated : &d_vctGroupBeginIndicesUpdated[d_vctRegroupingPlan[numGroup - 1]];
//
//    pGroupBeginIndicesUpdated[0] = d_vctGroupBeginIndices[numGroup];
//
//    int idx = 1;
//
//    ++pVal;
//    for (int i = 1; i < lengthGroup; ++i)
//    {
//        if (((*pVal) - (*(pVal - 1))) > d_bin_metrics)
//        {
//            pGroupBeginIndicesUpdated[idx] = pGroupBeginIndicesUpdated[0] + i;
//            ++idx;
//        }
//        ++pVal;
//
//    }
//}
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
    if (threadIdx.x >= lenGroup)
    {
        iarr[threadIdx.x] = 0;
    }
    else
    {
        // Get the value of the member using the offset       
        int quantSubGroups = (threadIdx.x == 0) ? 1 : 0;
       
        int stride = blockDim.x;
        for (int i = idx; i < lenGroup; i += stride)
        {            
            int indCur = iBeginGroupIndeces + i;
            // extract value
            const char* base = reinterpret_cast<const char*>(&d_arrCand[d_arrIndeces[indCur]]);
            // Get the value of the member using the offset
            int ivalCur = *reinterpret_cast<const int*>(base + member_offset);
            d_arrValues[indCur] = ivalCur;
            if (i == 0)
            {
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
    }
    __syncthreads();

    // Parallel reduction within the block
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
    __syncthreads();
}

//--------------------------------------------------------------------------------------------------------------
void print_fnc_grouping_output(const thrust::device_vector<Cand>& d_vctCandHeap,
    const int member_offset,
    const int& d_bin_metrics,
    const thrust::device_vector<int>& d_vctIndeces,
    const thrust::device_vector<int>& d_vctGroupBeginIndices) {
    // Print d_vctCandHeap
    std::cout << "d_vctCandHeap: ";
    thrust::host_vector<Cand> h_vctCandHeap = d_vctCandHeap;  // Transfer to host
    for (const auto& elem : h_vctCandHeap) {
        std::cout << "Cand(mt: " << elem.mt << ", mdt: " << elem.mdt
            << ", mwidth: " << elem.mwidth << ", msnr: " << elem.msnr << ") ";
    }
    std::cout << std::endl;

    // Print member_offset
    std::cout << "member_offset: " << member_offset << std::endl;

    // Print d_bin_metrics
    std::cout << "d_bin_metrics: " << d_bin_metrics << std::endl;

    // Print d_vctIndeces
    std::cout << "d_vctIndeces: ";
    thrust::host_vector<int> h_vctIndices = d_vctIndeces;  // Transfer to host
    for (const auto& elem : h_vctIndices) {
        std::cout << elem << " ";
    }
    std::cout << std::endl;

    // Print d_vctGroupBeginIndices
    std::cout << "d_vctGroupBeginIndices: ";
    thrust::host_vector<int> h_vctGroupBeginIndices = d_vctGroupBeginIndices;  // Transfer to host
    for (const auto& elem : h_vctGroupBeginIndices) {
        std::cout << elem << " ";
    }
    std::cout << std::endl;
}
//---------------------------------------------
void print_input_clusterization(const float* d_digitized_fdmt, const int Rows, const int Cols,
    const float& VAlTresh, const int WndWidth,
    const int* d_pbin_metrics, const std::string& filename) 
{
    // Allocate host memory to copy data from device
    float* h_digitized_fdmt = new float[Rows * Cols];
    float h_VAlTresh;
    int h_pbin_metrics[3];

    // Copy data from device to host
    cudaMemcpy(h_digitized_fdmt, d_digitized_fdmt, Rows * Cols * sizeof(float), cudaMemcpyDeviceToHost);
    checkCudaError1("Failed to copy d_digitized_fdmt from device to host");

    cudaMemcpy(&h_VAlTresh, &VAlTresh, sizeof(float), cudaMemcpyDeviceToHost);
    checkCudaError1("Failed to copy VAlTresh from device to host");

    cudaMemcpy(h_pbin_metrics, d_pbin_metrics, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    checkCudaError1("Failed to copy d_pbin_metrics from device to host");

    // Print the values
    std::cout << "d_digitized_fdmt:" << std::endl;
    for (int i = 0; i < Rows; ++i) {
        for (int j = 0; j < Cols; ++j) {
            std::cout << h_digitized_fdmt[i * Cols + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "VAlTresh: " << h_VAlTresh << std::endl;
    std::cout << "WndWidth: " << WndWidth << std::endl;

    std::cout << "d_pbin_metrics: ";
    for (int i = 0; i < 3; ++i) {
        std::cout << h_pbin_metrics[i] << " ";
    }
    std::cout << std::endl;

    // Clean up host memory
    delete[] h_digitized_fdmt;
}
//-------------------------------------------------------------------------------------------------------
void print_regrouping_input(int* d_vctValues,
    const int LEn_vctValues,
    const int* d_vctGroupBeginIndices,
    const int* d_vctRegroupingPlan,
    const int LEn_vctGroupBeginIndices)
{
    printDeviceArray(d_vctValues, LEn_vctValues, "d_vctValues");
    printDeviceArray(d_vctGroupBeginIndices, LEn_vctGroupBeginIndices, "d_vctGroupBeginIndices");
    printDeviceArray(d_vctRegroupingPlan, LEn_vctGroupBeginIndices, "d_vctRegroupingPlan");
}

int apb(int a, int b)
{
    return a + b;
}

