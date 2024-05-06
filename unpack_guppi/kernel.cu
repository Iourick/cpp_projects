
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cufft.h>
#include <iostream>
#include <complex>

int unpack_padd(int* arrRe, int* arrIm, int  nsamp,
    int nsub,
    int nbin,
    int noverlap,
    int   nfft, int* arrout)
{
    memset(arrout, 0, nfft * nsub * nbin * sizeof(int));
    for (int ifft = 0; ifft < nfft; ++ifft)
    {
        for (int isub = 0; isub < nsub; ++isub)
        {
            for (int ibin = 0; ibin < nbin; ++ibin)
            {
                int isamp = ibin + (nbin - 2 * noverlap) * ifft - noverlap;
                if ((isamp >= 0) && (isamp < nsamp))
                {
                    //cp[ifft, isub, ibin] = in_arr_re[idx2] + 1j * in_arr_im[idx2]
                    int idx2 = isub + nsub * isamp;
                    int iii = arrRe[idx2];
                    arrout[ifft * nbin * nsub + isub * nbin + ibin] = arrRe[idx2];
                }
            }
        }
    }
    return 1;
}
int calc_ChunkPolarizationComplexNumbers(int  m_nfft, int m_nchan, int m_nbin)
{
    return   m_nfft * m_nchan * m_nbin;
}

//---------------------------------------------------------------------------------
bool unpack_chunk_lofar(const long long LenChunk, const int Noverlap, int nbin, int nfft, int npol, int nchan
    , int* parrInput, std::complex<float>* arrOut)
{
    if (LenChunk != ((nbin - 2 * Noverlap) * nfft))
    {
        printf("LenChunk is not true");
        return false;
    }
    for (int k = 0; k < npol / 2; ++k)
    {
        std::complex<float>* pout_begin = &(arrOut)[nfft * nchan * nbin * k];
        int* arrRe = &parrInput[2 * k * nchan * LenChunk];
        int* arrIm = &parrInput[(1 + 2 * k) * nchan * LenChunk];

        for (int ifft = 0; ifft < nfft; ++ifft)
        {
            for (int isub = 0; isub < nchan; ++isub)
            {
                for (int ibin = 0; ibin < nbin; ++ibin)
                {
                    int isamp = ibin + (nbin - 2 * Noverlap) * ifft - Noverlap;
                    if ((isamp >= 0) && (isamp < LenChunk))
                    {
                        int idx2 = isub + nchan * isamp;

                        pout_begin[ifft * nbin * nchan + isub * nbin + ibin].real((float)arrRe[idx2]);
                        pout_begin[ifft * nbin * nchan + isub * nbin + ibin].imag((float)arrIm[idx2]);
                    }
                }
            }
        }
    }
    return true;
}




// 
__global__
void unpackInput_lofar(cufftComplex* pcmparrRawSignalCur,int* d_parrInput, const int NPolPh, const int LenChunk, const int nbin, const int  Noverlap)
{
    const int ibin = threadIdx.x + blockDim.x * blockIdx.x; // down channels, number of channel, inside fft
    if (ibin >= nbin)
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
//---------------------------------------------------------------------------------
bool unpack_chunk_guppi(const long long LenChunk, const int Noverlap, int nbin, int nfft, int npol, int nchan
    , int* parrInput, std::complex<float>* arrOut)
{
    if (LenChunk != ((nbin - 2 * Noverlap) * nfft))
    {
        printf("LenChunk is not true");
        return false;
    }
    for (int ichan = 0; ichan < nchan; ++ichan)
    {
        for (int ifft = 0; ifft < nfft; ++ifft)
        {
            for (int ibin = 0; ibin < nbin; ++ibin)
            {
                int isamp = ibin + (nbin - 2 * Noverlap) * ifft - Noverlap;
                for (int ipol = 0; ipol < npol / 2; ++ipol)
                {
                    int ind1 = ipol * nfft * nchan * nbin + ifft * nbin * nchan + ichan * nbin + ibin;
                    arrOut[ind1].real(0.0f);
                    arrOut[ind1].imag(0.0f);
                    if ((isamp >= 0) && (isamp < LenChunk))
                    {
                        int idx2 = ichan * LenChunk * npol + isamp * npol + ipol * 2;
                        arrOut[ind1].real((float)parrInput[idx2]);
                        arrOut[ind1].imag((float)parrInput[idx2 + 1]);
                    }
                }
            }
        }
    }

    return true;
}
//-----------------------------------------------------------------
//-----------------------------------------------------
//dim3 threads(256, 1, 1);
//dim3 blocks((nbin + threads.x - 1) / threads.x, nfft, nchan);
__global__
void unpack_chunk_guppi_gpu( int nsamp, int Noverlap, int nbin, int  nfft, int npol, int nchan
    , int* parrInput, cufftComplex*d_parrOut)
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




int main()
{
    std::cout << "Hello World!\n";
    int nbin =  6;
    int npol = 4;
    int  nchan =  3;
    const int Noverlap = 1;
    int nfft = 2;


    int nsamp = nfft * (nbin - 2 * Noverlap);

    int* parrInput = new int[nsamp * nchan * npol];

    for (int i = 0; i < nsamp * nchan * npol; ++i)
    {
        parrInput[i] = 1 + i;

    }


    int it = nfft * nchan * nbin * npol / 2;
    std::complex<float>* arrOut = new std::complex<float>[nfft * nchan * nbin * npol / 2];

    unpack_chunk_lofar(nsamp, Noverlap, nbin, nfft, npol, nchan, parrInput, arrOut);
    std::cout << std::endl;
    std::cout << "         INPUT" << std::endl;
    for (int ipol = 0; ipol < npol; ++ipol)
    {
        std::cout << "polarization No" << ipol << std::endl;
        for (int i = 0; i < nsamp; ++i)
        {
            for (int j = 0; j < nchan; ++j)
            {
                std::cout << parrInput[ipol * nsamp * nchan + i * nchan + j] << " ; ";
            }
            std::cout << std::endl;
        }
    }

    std::cout << "                         OUTPUT" << std::endl;
    for (int ipol = 0; ipol < npol / 2; ++ipol)
    {
        std::cout << "polarization No" << ipol << std::endl;
        for (int k = 0; k < nfft; ++k)
        {
            for (int i = 0; i < nchan; ++i)
            {
                for (int j = 0; j < nbin; ++j)
                {
                   std::cout << arrOut[ipol * nfft * nchan * nbin + k * nchan * nbin + i * nbin + j] << " ; ";
                }
                std::cout << std::endl;
            }
        }
    }
    std::cout << "Hello, world!" << std::endl;

    /// GUPPI
    std::cout << std::endl;
    std::cout << "    GUPPI     INPUT" << std::endl;
    for (int ichan = 0; ichan < nchan; ++ichan)
    {
        std::cout << "CHANEL No" << ichan << std::endl;
        for (int j = 0; j < npol * nsamp; ++j)
        {
            std::cout << parrInput[ichan * npol * nsamp + j] << " ; ";
        }
        std::cout << std::endl;
    }
    unpack_chunk_guppi(nsamp, Noverlap, nbin, nfft, npol, nchan
        , parrInput, arrOut);

    std::cout << "       GUPPI     CPU     OUTPUT" << std::endl;
    for (int ipol = 0; ipol < npol / 2; ++ipol)
    {
        std::cout << "polarization No" << ipol << std::endl;
        for (int k = 0; k < nfft; ++k)
        {
            for (int i = 0; i < nchan; ++i)
            {
                for (int j = 0; j < nbin; ++j)
                {
                    std::cout << arrOut[ipol * nfft * nchan * nbin + k * nchan * nbin + i * nbin + j] << " ; ";
                }
                std::cout << std::endl;
            }
        }
    }

    //-------------------   GPU GUPPI -------------------------------------------------------------------------------------
    int* d_parrInput = nullptr;
    cudaMalloc((void**)&d_parrInput, nsamp * nchan * npol * sizeof(int));
    cudaMemcpy(d_parrInput, parrInput, nsamp * nchan * npol * sizeof(int), cudaMemcpyHostToDevice);
    cufftComplex* d_pOut = nullptr;
    cudaMalloc((void**)&d_pOut, nfft * nchan * nbin * npol / 2 * sizeof(cufftComplex));

    dim3 threads(256, 1, 1);
    dim3 blocks((nbin + threads.x -1)/threads.x, nfft, nchan);
    unpack_chunk_guppi_gpu<<< blocks, threads>>>(nsamp, Noverlap, nbin, nfft, npol, nchan
        , parrInput, d_pOut);

    cudaMemcpy(arrOut, d_pOut, nfft * nchan * nbin * npol / 2 * sizeof(std::complex<float>), cudaMemcpyDeviceToHost);

    std::cout << "       GUPPI     GPU     OUTPUT" << std::endl;
    std::cout << "       GUPPI     GPU     OUTPUT" << std::endl;
    for (int ipol = 0; ipol < npol / 2; ++ipol)
    {
        std::cout << "polarization No" << ipol << std::endl;
        for (int k = 0; k < nfft; ++k)
        {
            for (int i = 0; i < nchan; ++i)
            {
                for (int j = 0; j < nbin; ++j)
                {
                    std::cout << arrOut[ipol * nfft * nchan * nbin + k * nchan * nbin + i * nbin + j] << " ; ";
                }
                std::cout << std::endl;
            }
        }
    }

    delete[]parrInput;
    delete[]arrOut;
    cudaFree(d_parrInput);
    cudaFree(d_pOut);

}
