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
#include "Chunk_v1_gpu.cuh"
#include "Chunk_fly_gpu.cuh"
#include "Chunk_v3_gpu.cuh"
#include "Chunk_v4_gpu.cuh"
#include "Chunk_py_gpu.cuh"
#include "npy.hpp"

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


bool  CSession_lofar_gpu::unpack_chunk(const long long LenChunk, const int Noverlap, inp_type_* d_parrInput, void* pcmparrRawSignalCur)
{
    //cudaDeviceSynchronize();

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus));
        return false;
    }
    //
    const dim3 block_size(512, 1, 1);
    const dim3 gridSize((m_nbin + block_size.x - 1) / block_size.x, m_header.m_nchan, m_nfft);

    unpackInput_lofar << < gridSize, block_size >> > ((cufftComplex*)pcmparrRawSignalCur, (inp_type_*)d_parrInput, m_header.m_npol / 2, LenChunk, m_nbin, Noverlap);


   /* int lenarr =  m_nbin * m_header.m_nchan * m_nfft;
     std::vector<std::complex<float>> data2(lenarr, 0);
    cudaMemcpy(data2.data(), pcmparrRawSignalCur, lenarr* sizeof(std::complex<float>),
        cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    std::array<long unsigned, 1> leshape127{ lenarr };
    npy::SaveArrayAsNumpy("inp.npy", false, leshape127.size(), leshape127.data(), data2);*/
    return true;
}
//--------------------------------------------------------------------------------------------
bool CSession_lofar_gpu::allocateInputMemory(void** d_pparrInput, const int QUantDownloadingBytesForChunk, void** pcmparrRawSignalCur
    , const int QUantChunkComplexNumbers)
{
    checkCudaErrors(cudaMalloc((void**)d_pparrInput, QUantDownloadingBytesForChunk * sizeof(char)));
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
    , const int noverlap
    , const float tsamp)
{
   
        CChunkB* chunk = new CChunk_fly_gpu(Fmin
            , Fmax
            , npol
            , nchan           
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
            , noverlap
           , tsamp);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus));
            return;
        }
    *ppchunk = chunk;
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus));
        return ;
    }
}

//-----------------------------------------------------------------------

__global__
void unpackInput_lofar(cufftComplex* pcmparrRawSignalCur, inp_type_* d_parrInput, const int NPolPh, const int LenChunk   , const int nbin, const int  Noverlap)
{
    const int ibin = threadIdx.x + blockDim.x * blockIdx.x; // down channels, number of channel, inside fft
    if (ibin >= nbin )
    {
        return;
    }
    const int  NChan = gridDim.y;
    const int nfft = gridDim.z;
    const int  isub = blockIdx.y;
    const int ifft = blockIdx.z;
    int idx = ifft * NChan * nbin + isub * nbin + ibin;
    int isamp = ibin + (nbin - 2 * Noverlap) * ifft - Noverlap;

    if ((isamp >= 0) && (isamp < LenChunk))
    {
        int idx_inp = isamp * NChan + isub;
        pcmparrRawSignalCur[idx].x = (float)d_parrInput[idx_inp];
        pcmparrRawSignalCur[idx].y = (float)d_parrInput[LenChunk * NChan + idx_inp];
        if (NPolPh == 2)
        {
            idx += nfft * NChan * nbin;
            idx_inp += 2 * NChan * LenChunk;
            pcmparrRawSignalCur[idx].x = (float)d_parrInput[idx_inp];
            pcmparrRawSignalCur[idx].y = (float)d_parrInput[LenChunk * NChan + idx_inp];
        }
    }
    else
    {
       
        pcmparrRawSignalCur[idx].x = 0.0f;
        pcmparrRawSignalCur[idx].y = 0.0f;
        if (NPolPh == 2)
        {
            idx += nfft * NChan * nbin;

            pcmparrRawSignalCur[idx].x = 0.0f;
            pcmparrRawSignalCur[idx].y = 0.0f;
        }
    }    
}

//-----------------------------------------------------------

size_t CSession_lofar_gpu::download_chunk(FILE** prb_File, char* d_parrInput, const long long QUantDownloadingBytes)
{
    const long long QUantDownloadingBytesForEachFile = QUantDownloadingBytes / m_header.m_npol;
    size_t sreturn = 0;
    char* parrInput = new char[QUantDownloadingBytes];
    for (int i = 0; i < 4; i++)
    {
        size_t nread = fread(&parrInput[QUantDownloadingBytesForEachFile * i], sizeof(char), QUantDownloadingBytesForEachFile, prb_File[i]);
        if (nread == 0)
        {
            delete[]parrInput;
            break;
        }
        sreturn += nread;
    }
    cudaMemcpy(d_parrInput, parrInput, QUantDownloadingBytes, cudaMemcpyHostToDevice);
    delete[]parrInput;
    return sreturn;
}



