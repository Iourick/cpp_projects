#include "Session_guppi_gpu.cuh"
#include <stdlib.h>



#include "helper_functions.h"
#include "helper_cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <complex>
#include <cufft.h>
#include "ChunkB.h"
#include "Chunk_gpu.cuh"
#include "Chunk_fly_gpu.cuh"



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

bool   CSession_guppi_gpu::unpack_chunk(const long long lenChunk, const int Noverlap, inp_type_* d_parrInput, void* pcmparrRawSignalCur)
{
    dim3 threads(256, 1, 1);
    dim3 blocks((m_nbin + threads.x - 1) / threads.x, m_nfft, m_header.m_nchan);
    unpack_chunk_guppi_gpu<<< blocks, threads>>>(lenChunk, Noverlap, m_nbin, m_nfft, m_header.m_npol, m_header.m_nchan
        , d_parrInput, (cufftComplex*)pcmparrRawSignalCur);
    return true;
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
        CChunkB* chunk  = new CChunk_fly_gpu(Fmin
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
    *ppchunk = chunk;
}

//-----------------------------------------------------------------------
//dim3 threads(256, 1, 1);
//dim3 blocks((nbin + threads.x - 1) / threads.x, nfft, nchan);
__global__
void unpack_chunk_guppi_gpu(int nsamp, int Noverlap, int nbin, int  nfft, int npol, int nchan
    , inp_type_* parrInput, cufftComplex* d_parrOut)
{
    const int ibin = threadIdx.x + blockDim.x * blockIdx.x;
    if (ibin >= nbin)
    {
        return;
    }
    const int ifft = blockIdx.y;
    const int ichan = blockIdx.z;
    int isamp = ibin + (nbin - 2 * Noverlap) * ifft - Noverlap;
    for (int ipol = 0; ipol < npol / 2; ++ipol)
    {
        int ind1 = ipol * nfft * nchan * nbin + ifft * nbin * nchan + ichan * nbin + ibin;
        d_parrOut[ind1].x = 0.0f;
        d_parrOut[ind1].y = 0.0f;
        if ((isamp >= 0) && (isamp < nsamp))
        {
            int idx2 = ichan * nsamp * npol + isamp * npol + ipol * 2;
            d_parrOut[ind1].x = (float)parrInput[idx2];
            d_parrOut[ind1].y = (float)parrInput[idx2 + 1];

        }
    }
}
//------------------------------------------
size_t  CSession_guppi_gpu::download_chunk(FILE** rb_File, char* d_parrInput, const long long QUantDownloadingBytes)
{
    char* parrInput = nullptr;
    if (!(parrInput = new char[QUantDownloadingBytes]))
    {
        return 0;
   }
    const long long position0 = ftell(*rb_File);
    long long quantDownloadingBytesPerChannel = QUantDownloadingBytes / m_header.m_nchan;
    long long quantTotalBytesPerChannel = m_header.m_nblocksize / m_header.m_nchan;
    //	
    char* p = (m_header.m_bSraightchannelOrder) ? parrInput : parrInput + (m_header.m_nchan - 1) * quantDownloadingBytesPerChannel;
    size_t sz_rez = 0;
    for (int i = 0; i < m_header.m_nchan; ++i)
    {
        // long long position1 = ftell(*rb_File);
        sz_rez += fread(p, sizeof(char), quantDownloadingBytesPerChannel, *rb_File);
        //  long long position2 = ftell(*rb_File);
        if (m_header.m_bSraightchannelOrder)
        {
            p += quantDownloadingBytesPerChannel;
        }
        else
        {
            p -= quantDownloadingBytesPerChannel;
        }

        if (i < m_header.m_nchan - 1)
        {
            fseek(*rb_File, quantTotalBytesPerChannel - quantDownloadingBytesPerChannel, SEEK_CUR);
        }

        // long long position3 = ftell(*rb_File);
    }
    // long long position4 = ftell(*rb_File);
    fseek(*rb_File, -(m_header.m_nchan - 1) * quantTotalBytesPerChannel, SEEK_CUR);
    // long long position5 = ftell(*rb_File);
    cudaMemcpy(d_parrInput, parrInput, QUantDownloadingBytes, cudaMemcpyHostToDevice);
    delete[] parrInput;
    return sz_rez;
}

