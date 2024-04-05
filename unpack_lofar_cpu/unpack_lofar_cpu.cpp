// unpack_lofar_cpu.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
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

bool unpack_chunk_(const long long LenChunk, const int Noverlap, int m_nbin, int m_nfft, int  m_nchan
    , int* d_parrInput, int* pcmparrRawSignalCur)
{ 
        for (int ifft = 0; ifft < m_nfft; ++ifft)
        {
            for (int isub = 0; isub < m_nchan; ++isub)
            {
               
                for (int ibin = 0; ibin < m_nbin; ++ibin)
                {                
                                   
                    pcmparrRawSignalCur[ifft * m_nchan * m_nbin + isub * m_nbin + ibin] = d_parrInput[ifft * m_nchan * (m_nbin - 2 * Noverlap) + ibin * m_nchan + isub];                       
                }
            }
        }   

    return 1;
}


int main()
{
    std::cout << "Hello World!\n";
    int m_nbin = 6;
    int m_npol = 4;
    int  m_nchan = 3;
    const int Noverlap = 1;
    int m_nfft = 1;
    const long long LenChunk = (m_nbin + ((m_nbin - 2 * Noverlap) * (m_nfft - 1)));
    int it = LenChunk * m_npol * m_nchan;
    int* d_parrInput = new int[LenChunk * m_npol  * m_nchan];
    for (int i = 0; i < it; ++i)
    {
        d_parrInput[i] = i + 1;
    }
    int* pcmparrRawSignalCur = new int[m_nfft * m_npol / 2 * m_nbin * m_nchan];
    int uuu = m_nfft * m_npol / 2 * m_nbin;
    ////-------------------------- No 2 var ---------------------------------------------------------------------
    std::cout << "INPUT MATRIX RE,  POL = 0!\n"<<std::endl;
    for (int i = 0; i < LenChunk; ++i)
    {
        for (int j = 0; j < m_nchan; ++j)
        {
            std::cout << d_parrInput[i * m_nchan + j] << " ; ";
        }
        std::cout << std::endl;
    }


    //unpack_chunk_(LenChunk, Noverlap, m_nbin, m_nfft, m_nchan , d_parrInput, pcmparrRawSignalCur);


    //std::cout << std::endl;
    //std::cout << "OUTPUT MATRIX RE,  POL = 0!\n";
    //std::cout << "*** IPOL = 0   **********" << std::endl;
    //std::cout << "******************************" << std::endl;

    //
    //    for (int k = 0; k < m_nfft; ++k)
    //    {
    //        for (int i = 0; i < m_nchan; ++i)
    //        {
    //            for (int j = 0; j < m_nbin; ++j)
    //            {
    //                std::cout << pcmparrRawSignalCur[ k * m_nchan * m_nbin + i * m_nbin + j] << " ; ";
    //            }
    //            std::cout << std::endl;
    //        }
    //        std::cout << "******************************" << std::endl;
    //    
    //    std::cout << "******************************" << std::endl;
    //    std::cout << "*** IPOL = 1   **********" << std::endl;
    //    std::cout << "******************************" << std::endl;
    //}

    //-------------------------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------------------------
    //--------  VAR No1                 -----------------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------------------------




    //unpack_chunk( LenChunk,  Noverlap, m_nbin,m_nfft, m_npol,  m_nchan,  d_parrInput,  pcmparrRawSignalCur);


    //std::cout << "INPUT MATRIX RE,  POL = 0!\n";
    //for (int i = 0; i < LenChunk; ++i)
    //{
    //    for (int j = 0; j < m_nchan; ++j)
    //    {
    //        std::cout << d_parrInput[i * m_nchan + j] << " ; ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;
    //std::cout << "Hello, world!" << std::endl;
    //std::cout << "INPUT MATRIX RE, POL = 1!\n";
    //for (int i = 0; i < LenChunk; ++i)
    //{
    //    for (int j = 0; j < m_nchan; ++j)
    //    {
    //        std::cout << d_parrInput[LenChunk * m_nchan *2+ i * m_nchan + j] << " ; ";
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << std::endl;
    //std::cout << "OUTPUT MATRIX RE,  POL = 0!\n";
    //std::cout << "*** IPOL = 0   **********" << std::endl;
    //std::cout << "******************************" << std::endl;
    //
    //for (int ipol = 0; ipol < m_npol / 2; ++ipol)
    //{
    //    for (int k = 0; k < m_nfft; ++k)
    //    {
    //        for (int i = 0; i < m_nchan; ++i)
    //        {
    //            for (int j = 0; j < m_nbin; ++j)
    //            {
    //                std::cout << pcmparrRawSignalCur[ipol * m_nfft * m_nchan * m_nbin  +   k * m_nchan * m_nbin + i * m_nbin + j] << " ; ";
    //            }
    //            std::cout << std::endl;
    //        }
    //        std::cout << "******************************" << std::endl;
    //    }
    //    std::cout << "******************************" << std::endl;
    //    std::cout << "*** IPOL = 1   **********" << std::endl;
    //    std::cout << "******************************" << std::endl;
    //}
    delete[]d_parrInput;
    delete [] pcmparrRawSignalCur;
    //----------------  PRAVIR   ------------------------------------
    //----------------  PRAVIR   ------------------------------------
    //----------------  PRAVIR   ------------------------------------
    //----------------  PRAVIR   ------------------------------------
    //----------------  PRAVIR   ------------------------------------
    int nsamp = m_nfft * (m_nbin - 2 * Noverlap);
    int* arrRe = new int[nsamp * m_nchan];
    int* arrIm = new int[nsamp * m_nchan];
    for (int i = 0; i < nsamp * m_nchan; ++i)
    {
        arrRe[i] = 1 + i;
        arrIm[i] = nsamp * m_nchan + 1 + i;
    }
    int* pio = new int[m_nfft* m_nchan* m_nbin];
    unpack_padd(arrRe, arrIm, nsamp,
        m_nchan,
        m_nbin,
        Noverlap,
        m_nfft, pio);
    std::cout << std::endl;
    std::cout << std::endl;
    for (int i = 0; i < nsamp; ++i)
    {
        for (int j = 0; j < m_nchan; ++j)
        {
            std::cout << arrRe[i * m_nchan + j] << " ; ";
        }
        std::cout << std::endl;
    }
    std::cout << "Hello, world!" << std::endl;
    for (int i = 0; i < m_nchan; ++i)
    {
        for (int j = 0; j < m_nbin; ++j)
        {
            std::cout << pio[i * m_nbin + j] << " ; ";
        }
        std::cout << std::endl;
    }
    std::cout << "Hello, world!" << std::endl;
   /* std::cout << "2 matrice!" << std::endl;
    for (int i = 0; i < m_nchan; ++i)
    {
        for (int j = 0; j < m_nbin; ++j)
        {
            std::cout << pio[m_nchan * m_nbin + i * m_nchan + j] << " ; ";
        }
        std::cout << std::endl;
    }*/

    delete[]arrRe;
    delete[]arrIm;
    delete[]pio;
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
