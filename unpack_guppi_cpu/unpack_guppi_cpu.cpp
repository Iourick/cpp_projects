// unpack_lofar_cpu.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <complex>
//
//@njit
//def unpack_and_padd(
//    in_arr_re: np.ndarray,
//    in_arr_im : np.ndarray,
//    nsamp : int,
//    nsub : int,
//    nbin : int,
//    noverlap : int,
//    nfft : int = 100,
//    )->np.ndarray:

//cp = np.zeros((nfft, nsub, nbin), dtype = np.complex64)
//for ifft in range(nfft) :
//    for isub in range(nsub) :
//        for ibin in range(nbin) :
//            # Calculate input sample index considering overlap.
//            isamp = ibin + (nbin - 2 * noverlap) * ifft - noverlap
//            if 0 <= isamp < nsamp:
//# Calculate the correct index in the input array.
//idx2 = isub + nsub * isamp
//cp[ifft, isub, ibin] = in_arr_re[idx2] + 1j * in_arr_im[idx2]
//return cp
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
bool unpack_chunk(const long long LenChunk, const int Noverlap, int m_nbin, int m_nfft, int m_npol, int  m_nchan
    , int* d_parrInput, int* pcmparrRawSignalCur)
{
    if (LenChunk != (m_nbin + ((m_nbin - 2 * Noverlap) * (m_nfft - 1))))
    {
        printf("LenChunk is not true");
        return false;
    }

    for (int k = 0; k < m_npol / 2; ++k)
    {
        int* pout_begin = &(pcmparrRawSignalCur)[m_nfft * m_nchan * m_nbin * k];
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
                    pout_begin[ifft * m_nchan * m_nbin + isub * m_nbin + ibin] = pinpBegin[ibin * m_nchan + isub];
                    int ia = pout_begin1[isub * m_nbin + ibin];
                    int tt = 0;
                    //printf(" j = %i", pout_begin1 - pout_begin);
                }
            }
        }
    }

    return 1;
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
                        arrOut[ind1].imag((float)parrInput[idx2 +1]);
                    }
                }
            }
        }
    }
    
    return true;
}





int main()
{   
    
    std::cout << "Hello World!\n";
    int nbin = 6;
    int npol = 4;
    int  nchan = 3;
    const int Noverlap = 1;
    int nfft = 2;
    
    
    int nsamp = nfft * (nbin - 2 * Noverlap) ;

    int* parrInput = new int[nsamp * nchan * npol];
    
    for (int i = 0; i < nsamp * nchan * npol; ++i)
    {
        parrInput[i] = 1 + i;
       
    }


    int it = nfft * nchan * nbin * npol / 2;
    std::complex<float>* arrOut = new std::complex<float>[nfft * nchan * nbin * npol / 2];
   
    unpack_chunk_lofar(nsamp,Noverlap, nbin,  nfft,  npol,  nchan, parrInput, arrOut);
    std::cout << std::endl;
    std::cout <<"         INPUT"<< std::endl;
    for (int ipol = 0; ipol < npol ; ++ipol)
    {
        std::cout << "polarization No" <<ipol<< std::endl;
        for (int i = 0; i < nsamp; ++i)
        {
            for (int j = 0; j < nchan; ++j)
            {
                std::cout << parrInput[ipol  * nsamp * nchan + i * nchan + j] << " ; ";
            }
            std::cout << std::endl;
        }
    }
  
    std::cout << "                         OUTPUT" << std::endl;
    for (int ipol = 0; ipol < npol / 2; ++ipol)
    {
        std::cout << "polarization No"<<ipol << std::endl;
        for (int k = 0; k < nfft; ++k)
        {
            for (int i = 0; i < nchan; ++i)
            {
                for (int j = 0; j < nbin; ++j)
                {
                    std::cout << arrOut[ipol  * nfft  * nchan * nbin + k * nchan * nbin + i * nbin + j] << " ; ";
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
            std::cout << parrInput[ichan * npol * nsamp +j] << " ; ";
        }
        std::cout << std::endl;
    }
    unpack_chunk_guppi(nsamp, Noverlap, nbin, nfft, npol, nchan
        , parrInput, arrOut);

    std::cout << "       GUPPI          OUTPUT" << std::endl;
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
    
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
