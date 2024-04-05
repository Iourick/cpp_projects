#include "Session_lofar_gpu.cuh"
#include <stdlib.h>
#include "Fragment.h"

#include "helper_cuda.h"
#include "yr_cart.h"
#include "helper_functions.h"
#include "helper_cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include "Chunk.cuh"
#include <complex>
#include <cufft.h>
#include "ChunkB.h"
#include "Chunk_gpu.cuh"

cudaError_t cudaStatus;

//-------------------------------------------
CSession_lofar_gpu::CSession_lofar_gpu() :CSession_lofar()
{
}

//--------------------------------------------
CSession_lofar_gpu::CSession_lofar_gpu(const  CSession_lofar_gpu& R) :CSession_lofar(R)
{
}

//-------------------------------------------
CSession_lofar_gpu& CSession_lofar_gpu::operator=(const CSession_lofar_gpu& R)
{
    if (this == &R)
    {
        return *this;
    }
    CSession_lofar:: operator= (R);

    return *this;
}

//--------------------------------- 
CSession_lofar_gpu::CSession_lofar_gpu(const char* strGuppiPath, const char* strOutPutPath, const float t_p
    , const double d_min, const double d_max, const float sigma_bound, const int length_sum_wnd, const int nbin, const int nfft)
    :CSession_lofar(strGuppiPath, strOutPutPath,  t_p  ,  d_min,  d_max, sigma_bound, length_sum_wnd,  nbin,  nfft)
{
}
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------


bool  CSession_lofar_gpu::unpack_chunk(const long long lenChunk, const int j, inp_type_* d_parrInput, void* pcmparrRawSignalCur)
{
    cudaDeviceSynchronize();

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus));
        // Handle the error appropriately
    }
    //
    const dim3 block_size(1024, 1, 1);
    const dim3 gridSize((lenChunk + block_size.x - 1) / block_size.x, 1, 1);

    unpackInput_L <<< gridSize, block_size >> > ((cufftComplex * )pcmparrRawSignalCur, (inp_type_*)d_parrInput, lenChunk, m_header.m_nchan, m_header.m_npol);
    return true;
}
//--------------------------------------------------------------------------------------------
bool CSession_lofar_gpu::allocateInputMemory(void** d_pparrInput, const int QUantDownloadingBytesForChunk, void** pcmparrRawSignalCur
    , const int QUantChunkComplexNumbers)
{
    checkCudaErrors(cudaMallocManaged((void**)d_pparrInput, QUantDownloadingBytesForChunk * sizeof(char)));
    checkCudaErrors(cudaMalloc((void**)pcmparrRawSignalCur, QUantChunkComplexNumbers * sizeof(cufftComplex)));
    return true;
}
//------------------------------------------------------------------------------------
void CSession_lofar_gpu::freeInputMemory(void* parrInput, void* pcmparrRawSignalCur)
{
    checkCudaErrors(cudaFree(parrInput));
    checkCudaErrors(cudaFree(pcmparrRawSignalCur));
}
//----------------------------------------------------------------------------
void CSession_lofar_gpu::createChunk(CChunkB** ppchunk
    , const float Fmin
    , const float Fmax
    , const int npol
    , const int nchan
    , const unsigned int lenChunk
    , const unsigned int len_sft
    , const int Block_id
    , const int Chunk_id
    , const double d_max
    , const double d_min
    , const int ncoherent
    , const float sigma_bound
    , const int length_sum_wnd
    , const int nbin
    , const int nfft
    , const int noverlap)
{
   
        CChunk_gpu* chunk = new CChunk_gpu(Fmin
            , Fmax
            , npol
            , nchan
            , lenChunk
            , len_sft
            , Block_id
            , Chunk_id
            , d_max
            , d_min
            , ncoherent
            , sigma_bound
            , length_sum_wnd
            , nbin
            , nfft
            , noverlap);
    *ppchunk = chunk;
}

//-----------------------------------------------------------------------

__global__
void unpackInput_L(cufftComplex* pcmparrRawSignalCur, inp_type_* d_parrInput, const int  lenChunk
    , const int  NChan, const int  npol)
{
    const int itime = threadIdx.x + blockDim.x * blockIdx.x; // down channels, number of channel
    if (itime >= lenChunk)
    {
        return;
    }
    int numColInp0 = itime * NChan;

    int colsInp = lenChunk * NChan;


    for (int i = 0; i < NChan; ++i)
    {
        for (int j = 0; j < npol / 2; ++j)
        {
            pcmparrRawSignalCur[(i * npol / 2 + j) * lenChunk + itime].x
                = (float)d_parrInput[j * npol / 2 * colsInp + numColInp0 + i];
            pcmparrRawSignalCur[(i * npol / 2 + j) * lenChunk + itime].y
                = (float)d_parrInput[(1 + j * npol / 2) * colsInp + numColInp0 + i];
        }
    }

}
//