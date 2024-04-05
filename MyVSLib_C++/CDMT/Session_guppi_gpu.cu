#include "Session_guppi_gpu.cuh"
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



//-------------------------------------------
CSession_guppi_gpu::CSession_guppi_gpu() :CSession_guppi()
{
}

//--------------------------------------------
CSession_guppi_gpu::CSession_guppi_gpu(const  CSession_guppi_gpu& R) :CSession_guppi(R)
{
}

//-------------------------------------------
CSession_guppi_gpu& CSession_guppi_gpu::operator=(const CSession_guppi_gpu& R)
{
    if (this == &R)
    {
        return *this;
    }
    CSession_guppi:: operator= (R);

    return *this;
}

//--------------------------------- 
CSession_guppi_gpu::CSession_guppi_gpu(const char* strGuppiPath, const char* strOutPutPath, const float t_p
    , const double d_min, const double d_max, const float sigma_bound, const int length_sum_wnd, const int nbin, const int nfft)
    :CSession_guppi(strGuppiPath, strOutPutPath,  t_p  ,  d_min,  d_max, sigma_bound, length_sum_wnd,  nbin,  nfft)
{
}
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------


bool   CSession_guppi_gpu::unpack_chunk(const long long lenChunk, const int j, inp_type_* d_parrInput, void* pcmparrRawSignalCur)
{
   // code !!!!!
    return false;
}
//--------------------------------------------------------------------------------------------
bool CSession_guppi_gpu::allocateInputMemory(void** d_pparrInput, const int QUantDownloadingBytesForChunk, void** pcmparrRawSignalCur
    , const int QUantChunkComplexNumbers)
{
    checkCudaErrors(cudaMallocManaged((void**)d_pparrInput, QUantDownloadingBytesForChunk * sizeof(char)));
    checkCudaErrors(cudaMalloc((void**)pcmparrRawSignalCur, QUantChunkComplexNumbers * sizeof(cufftComplex)));
    return true;
}
//------------------------------------------------------------------------------------
void CSession_guppi_gpu::freeInputMemory(void* parrInput, void* pcmparrRawSignalCur)
{
    checkCudaErrors(cudaFree(parrInput));
    checkCudaErrors(cudaFree(pcmparrRawSignalCur));
}
//----------------------------------------------------------------------------
void CSession_guppi_gpu::createChunk(CChunkB** ppchunk
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

