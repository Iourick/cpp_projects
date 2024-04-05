#include "Session_lofar_cpu.h"
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
#include "Session_lofar.h"



//-------------------------------------------
CSession_lofar_cpu::CSession_lofar_cpu() :CSession_lofar()
{
}

//--------------------------------------------
CSession_lofar_cpu::CSession_lofar_cpu(const  CSession_lofar_cpu& R) :CSession_lofar(R)
{
}

//-------------------------------------------
CSession_lofar_cpu& CSession_lofar_cpu::operator=(const CSession_lofar_cpu& R)
{
    if (this == &R)
    {
        return *this;
    }
    CSession_lofar:: operator= (R);

    return *this;
}

//--------------------------------- 
CSession_lofar_cpu::CSession_lofar_cpu(const char* strGuppiPath, const char* strOutPutPath, const float t_p
    , const double d_min, const double d_max, const float sigma_bound, const int length_sum_wnd, const int nbin, const int nfft)
    :CSession_lofar(strGuppiPath, strOutPutPath, t_p
        , d_min, d_max,  sigma_bound, length_sum_wnd, nbin, nfft)
{
}
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------

void CSession_lofar_cpu::freeInputMemory(void* parrInput, void* pcmparrRawSignalCur)
{
    free(parrInput);
    fftw_free(pcmparrRawSignalCur);
}
//---------------------------------------------------------------------------------
/*bool unpack_chunk(const long long LenChunk, const int Noverlap, int m_nbin, int m_nfft, int m_npol, int  m_nchan
    , int* d_parrInput, int* pcmparrRawSignalCur)
{
    if (LenChunk != (m_nbin + ((m_nbin - 2 * Noverlap) * (m_nfft - 1))))
    {
        printf("LenChunk is not true");
        return false;
    }
    const long long num_chun = calc_ChunkPolarizationComplexNumbers(m_nfft, m_nchan, m_nbin);
    for (int k = 0; k < m_npol / 2; ++k)
    {
        int* pout_begin = &(pcmparrRawSignalCur)[m_nfft * m_nchan* m_nbin * k];
        int* pinpBegin0 = &d_parrInput[2 * k * LenChunk * m_nchan];
        int* pinp1 = &d_parrInput[(1 + 2 * k) * LenChunk * m_nchan];
        for (int ifft = 0; ifft < m_nfft; ++ifft)
        {
            for (int isub = 0; isub < m_nchan; ++isub)
            {
                int* pout_begin1 = pout_begin + ifft * m_nchan * m_nbin;
                for (int ibin = 0; ibin < m_nbin; ++ibin)
                {
                    int iout = ifft * m_nchan * m_nbin + isub * m_nbin + ibin;
                    int* pinpBegin = pinpBegin0 + ifft * m_nchan * (m_nbin - 2 * Noverlap);
                    // int iinp = iout - ifft * m_nchan * 2 * Noverlap;
                    pout_begin[ifft * m_nchan * m_nbin + isub * m_nbin + ibin ] = pinpBegin[ibin * m_nchan + isub];
                    int ia = pout_begin1[isub * m_nbin + ibin];
                    int tt = 0;
                    //printf(" j = %i", pout_begin1 - pout_begin);
                }
            }
        }
    }

    return 1;
}*/

bool CSession_lofar_cpu::unpack_chunk(const long long LenChunk, const int Noverlap
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
        inp_type_* arrIm = &d_parrInput[(1 +2 * k) * m_header.m_nchan * LenChunk];

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
                        int iii = arrRe[idx2];
                        pout_begin[ifft * m_nbin * m_header.m_nchan + isub * m_nbin + ibin][0] = (inp_type_)arrRe[idx2];
                        pout_begin[ifft * m_nbin * m_header.m_nchan + isub * m_nbin + ibin][1] = (inp_type_)arrIm[idx2];
                    }
                }
            }
        }
    }
    return true;
}

//----------------------------------------------------------------------
bool CSession_lofar_cpu::allocateInputMemory(void** parrInput, const int QUantDownloadingBytesForChunk, void** pcmparrRawSignalCur
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

void CSession_lofar_cpu::createChunk(CChunkB** ppchunk
    ,const float Fmin
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
        CChunk_cpu* chunk  = new CChunk_cpu(Fmin
        ,  Fmax
        , npol
        ,  nchan
        ,  lenChunk
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
        , noverlap);
        *ppchunk = chunk;
}
