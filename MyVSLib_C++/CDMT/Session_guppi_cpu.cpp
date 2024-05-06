#include "Session_guppi_cpu.h"
#include <string>
#include "stdio.h"
#include <iostream>

#include "OutChunk.h"


#include <stdlib.h>
#include "Fragment.h"
#include <fftw3.h>
#include <hdf5.h> 
#include <complex>
#include "yr_cart.h"
#include "Chunk_cpu.h"
#include <complex>
#include "Session_guppi.h"



//-------------------------------------------
CSession_guppi_cpu::CSession_guppi_cpu() :CSession_guppi()
{
}

//--------------------------------------------
CSession_guppi_cpu::CSession_guppi_cpu(const  CSession_guppi_cpu& R) :CSession_guppi(R)
{
}

//-------------------------------------------
CSession_guppi_cpu& CSession_guppi_cpu::operator=(const CSession_guppi_cpu& R)
{
    if (this == &R)
    {
        return *this;
    }
    CSession_guppi:: operator= (R);

    return *this;
}

//--------------------------------- 
CSession_guppi_cpu::CSession_guppi_cpu(const char* strGuppiPath, const char* strOutPutPath, const float t_p
    , const double d_min, const double d_max, const float sigma_bound, const int length_sum_wnd, const int nbin, const int nfft)
    :CSession_guppi(strGuppiPath, strOutPutPath, t_p
        , d_min, d_max,  sigma_bound, length_sum_wnd, nbin, nfft)
{
}
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------

void CSession_guppi_cpu::freeInputMemory(void* parrInput, void* pcmparrRawSignalCur)
{
    free(parrInput);
    fftw_free(pcmparrRawSignalCur);
}

//----------------------------------------------------------------------
bool CSession_guppi_cpu::allocateInputMemory(void** parrInput, const int QUantDownloadingBytesForChunk, void** pcmparrRawSignalCur
    , const int QUantChunkComplexNumbers) 
{
     if (!((*parrInput) = (void*)malloc(QUantDownloadingBytesForChunk)))
     {
         printf("Can't allocate memory for parrInput");
         return false;
     }
     if (!((*pcmparrRawSignalCur) = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * QUantChunkComplexNumbers)))
     {
         printf("Can't allocate memory for pcmparrRawSignalCur");
         free((*parrInput));
         return false;
     }
    return true;
}
//----------------------------------------------------------

void CSession_guppi_cpu::createChunk(CChunkB** ppchunk
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
        CChunk_cpu* chunk  = new CChunk_cpu(Fmin
        ,  Fmax
        , npol
        ,  nchan        
        ,  len_sft
        , Block_id
        , Chunk_id
        ,  d_max
        , d_min
        ,  ncoherent
        , sigma_bound
        , length_sum_wnd
        , nbin
        , nfft
        , noverlap
        , tsamp);
        *ppchunk = chunk;
}

//---------------------------------------------------------------------------------
bool CSession_guppi_cpu:: unpack_chunk(const long long LenChunk, const int Noverlap
    , inp_type_* parrInput, void* arrOut)
{
    if (LenChunk != ((m_nbin - 2 * Noverlap) * m_nfft))
    {
        printf("LenChunk is not true");
        return false;
    }
    fftwf_complex*ptr = (fftwf_complex*)arrOut;
    for (int ichan = 0; ichan < m_header.m_nchan; ++ichan)
    {
        for (int ifft = 0; ifft < m_nfft; ++ifft)
        {
            for (int ibin = 0; ibin < m_nbin; ++ibin)
            {
                int isamp = ibin + (m_nbin - 2 * Noverlap) * ifft - Noverlap;
                for (int ipol = 0; ipol < m_header.m_npol / 2; ++ipol)
                {
                    int ind1 = ipol * m_nfft * m_header.m_nchan * m_nbin + ifft * m_nbin * m_header.m_nchan + ichan * m_nbin + ibin;
                    ptr[ind1][0] = 0.0f;
                    ptr[ind1][1] = 0.0f;
                    
                    if ((isamp >= 0) && (isamp < LenChunk))
                    {
                        int idx2 = ichan * LenChunk * m_header.m_npol + isamp * m_header.m_npol + ipol * 2;
                        ptr[ind1][0] = (float)parrInput[idx2];
                        ptr[ind1][1] = (float)parrInput[idx2 + 1];
                        
                    }
                }
            }
        }
    }

    return true;
}

