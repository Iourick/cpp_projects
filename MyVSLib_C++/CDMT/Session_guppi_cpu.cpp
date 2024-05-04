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
bool CSession_guppi_cpu::unpack_chunk(const long long LenChunk, const int Noverlap
    , inp_type_* d_parrInput, void* pcmparrRawSignalCur)
{
    if (LenChunk != ((m_nbin - 2 * Noverlap) * m_nfft))
    {
        printf("LenChunk is not true");
        return false;
    }
    for (int k = 0; k < m_header.m_npol / 2; ++k)
    {
        fftwf_complex* pout_begin = &((fftwf_complex*)pcmparrRawSignalCur)[m_nfft * m_header.m_nchan * m_nbin * k];
        inp_type_* arrRe = &d_parrInput[2 * k * m_header.m_nchan * LenChunk];
        inp_type_* arrIm = &d_parrInput[(1 + 2 * k) * m_header.m_nchan * LenChunk];

        for (int ifft = 0; ifft < m_nfft; ++ifft)
        {
            for (int isub = 0; isub < m_header.m_nchan; ++isub)
            {
                for (int ibin = 0; ibin < m_nbin; ++ibin)
                {
                    int isamp = ibin + (m_nbin - 2 * Noverlap) * ifft - Noverlap;
                    if ((isamp >= 0) && (isamp < LenChunk))
                    {
                        int idx2 = isub + m_header.m_nchan * isamp;
                        pout_begin[ifft * m_nbin * m_header.m_nchan + isub * m_nbin + ibin][0] = (inp_type_)arrRe[idx2];
                        pout_begin[ifft * m_nbin * m_header.m_nchan + isub * m_nbin + ibin][1] = (inp_type_)arrIm[idx2];
                    }
                }
            }
        }
    }
    return true;
}

