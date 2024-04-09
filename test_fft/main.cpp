#include "main.h"
#include <iostream>
#include <complex>
#include <fftw3.h>

#include <fstream>

#include <array>
#include <string>

#include <vector>
#include <cstdlib> // For random value generation
#include <ctime>   // For seeding the random number generator
#include "npy.hpp"
#include <algorithm> 
#include "Class.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <omp.h>
#include <chrono>
#define _CRT_SECURE_NO_WARNINGS
using namespace std;
////--------------------------------------------------------------
//template <typename T>
//void roll_(T* arr, const int lenarr, const int ishift)
//{
//    if (ishift == 0)
//    {
//        return;
//    }
//    T* arr0 = (T*)malloc(ishift * sizeof(T));
//    T* arr1 = (T*)malloc((lenarr - ishift) * sizeof(T));
//    memcpy(arr0, arr + lenarr - 1 - ishift, ishift * sizeof(T));
//    memcpy(arr1, arr, (lenarr - ishift) * sizeof(T));
//    memcpy(arr, arr0, ishift * sizeof(T));
//    memcpy(arr + ishift, arr1, (lenarr - ishift) * sizeof(T));
//    free(arr0);
//    free(arr1);
//}

int main(int argc, char** argv)
{  
    const int leng = 11;
    int iarr[11];
    float arr[11];

    // Fill the array with natural numbers
    for (int i = 0; i < leng; ++i) {
        arr[i] = i + 1;
        iarr[i] = i + 1;
    }

    
    Class::roll_(arr, 11, 1);
    std::cout << "Array of natural numbers arr:" << std::endl;
    for (int i = 0; i < leng; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
    Class::roll_(iarr, 11, 1);
    std::cout << "Array of natural numbers iarr:" << std::endl;
    for (int i = 0; i < leng; ++i) {
        std::cout << iarr[i] << " ";
    }
    std::cout << std::endl;
    // ----------------------    TEST of CHIRP CHANNEL ---------------------------------------
    ////1.
    //int mbin = 10;
    //std::vector<double> arr_bin_freqs(mbin-1); // Create a vector with size mbin - 1

    //// Populate the vector
    //for (int i = 0; i < mbin - 1; ++i) {
    //    arr_bin_freqs[i] = i + 1.0;
    //}

    //// Print the elements of the vector
    //std::cout << "arr_bin_freqs: ";
    //for (int i = 0; i < mbin - 1; ++i) {
    //    std::cout << arr_bin_freqs[i] << " ";
    //}
    //std::cout << std::endl;
    ////!1
    //

    //// 2.
   
    //int m_nchan = 5;
    //int m_len_sft = 4;
    //std::vector<int> arr_freqs_chan(m_nchan * m_len_sft); // Create a vector with size d0 * d1

    //// Populate the vector
    //int counter = 1;
    //for (int i = 0; i < m_nchan; ++i) {
    //    for (int j = 0; j < m_len_sft; ++j) {
    //        arr_freqs_chan[i * m_len_sft + j] = counter++;
    //    }
    //}

    //// Print the elements of the vector
    //std::cout << "freq_chan: ";
    //for (int i = 0; i < m_nchan * m_len_sft; ++i) {
    //    std::cout << arr_freqs_chan[i] << " ";
    //}
    //std::cout << std::endl;
    ////2!

    //// 3
    //double start0 = 0.25;
    //double stop0 = 0.25 *3;
    //double step0 = 0.25;
    //int size0 = static_cast<int>((stop0 - start0 +0.000001) / step0);
    //std::vector<double> parr_coh_dm(size0); // Create a vector with appropriate size
    //int ndm = parr_coh_dm.size();
    //// Populate the vector
    //for (int i = 0; i < size0; ++i) {
    //    parr_coh_dm[i] = start0 + i * step0;
    //}

    //// Print the elements of the vector
    //std::cout << " parr_coh_dm: ";
    //for (int i = 0; i < size0; ++i) {
    //    std::cout << parr_coh_dm[i] << " ";
    //}
    //std::cout << std::endl;
    ////3!

    //// 4.
    //// 5 phase_delay
    //double* pcoh_dm = parr_coh_dm.data();
    //double* parr_bin_freqs = arr_bin_freqs.data();
    //double qx = parr_coh_dm[1] * (arr_bin_freqs[1] / arr_freqs_chan[1 * m_len_sft +1]) * (arr_bin_freqs[1] / arr_freqs_chan[1 * m_len_sft + 1])
    //    / (arr_freqs_chan[1 * m_len_sft + 1] + arr_bin_freqs[1]);
    //std::vector<double> arr_phase_delay(ndm * m_nchan * m_len_sft * (mbin-1));
    //for (int idm = 0; idm < ndm; ++idm)
    //    for (int ichan = 0; ichan < m_nchan; ++ichan)
    //    {
    //        for (int isft = 0; isft < m_len_sft; ++isft)
    //        {
    //            for (int ibin = 0; ibin < mbin-1; ++ibin)
    //            {
    //                arr_phase_delay[idm * m_nchan * m_len_sft * (mbin -1) + ichan * m_len_sft * (mbin -1) + isft * (mbin -1) + ibin]
    //                    = parr_coh_dm[idm] * (arr_bin_freqs[ibin] / arr_freqs_chan[ichan * m_len_sft + isft]) * (arr_bin_freqs[ibin] / arr_freqs_chan[ichan * m_len_sft + isft])
    //                    / (arr_freqs_chan[ichan * m_len_sft + isft] + arr_bin_freqs[ibin]);
    //                if ((idm == 1) && (ichan == 1) && (isft == 1) && (ibin == 1))
    //                {
    //                    double q = parr_coh_dm[idm] * (arr_bin_freqs[ibin] / arr_freqs_chan[ichan * m_len_sft + isft]) * (arr_bin_freqs[ibin] / arr_freqs_chan[ichan * m_len_sft + isft])
    //                        / (arr_freqs_chan[ichan * m_len_sft + isft] + arr_bin_freqs[ibin]);
    //                    double xx = arr_phase_delay[idm * m_nchan * m_len_sft * (mbin - 1) + ichan * m_len_sft * (mbin - 1) + isft * (mbin - 1) + ibin];
    //                    int yy = 0;
    //                }
    //            }
    //        }
    //    }

    //int jdm = 1;
    //int jchan = 1;
    //int jsft = 1;
    //double* parr = arr_phase_delay.data() + jdm * m_nchan * m_len_sft * (mbin - 1) + jchan * m_len_sft * (mbin - 1) + jsft * (mbin - 1);
    //std::cout << std::endl;
    //std::cout <<"arr_phase_delay"<< std::endl;
    //
    //    for (int ibin = 0; ibin < mbin -1; ++ibin)
    //    {
    //        std::cout << parr[ibin] << " ;  ";
    //    }
    //    std::cout << std::endl;
 


    //// 5!



    //// 6
    //double start1 = 10;
    //double stop1 = 101;
    //double step1 = 10;

    //int size1 = (stop1 - start1 +0.00001) / step1;

    //std::vector<double> taper(size1); // Create a vector with appropriate size

    //// Populate the vector
    //for (int i = 0; i < size1; ++i) {
    //    taper[i] = start1 + i * step1;
    //}

    //// Print the elements of the vector
    //std::cout << "taper: ";
    //for (int i = 0; i < size1; ++i) {
    //    std::cout << taper[i] << " ";
    //}
    //// 6!
    //std::vector < std::complex<double>>parr_dc(ndm * m_nchan * m_len_sft * (mbin - 1));
    //std::complex<double>minus_i1(0.0, -1.0);
    //for (int idm = 0; idm < ndm; ++idm)
    //    for (int ichan = 0; ichan < m_nchan; ++ichan)
    //    {
    //        for (int isft = 0; isft < m_len_sft; ++isft)
    //        {
    //            for (int ibin = 0; ibin < mbin-1; ++ibin)
    //            {
    //                (parr_dc)[idm * m_nchan * m_len_sft * (mbin - 1) + ichan * m_len_sft * (mbin - 1) + isft * (mbin - 1) + ibin]
    //                    =( std::exp(2.0 * minus_i1 * M_PI * arr_phase_delay[idm * m_nchan * m_len_sft * (mbin - 1) + ichan * m_len_sft * (mbin - 1) + isft * (mbin - 1) + ibin])) * taper[ibin];
    //               

    //            }
    //        }
    //    }
    //// 6!

    //int num0 =1;
    //int num1 = 1;
    //std::complex<double>* pcmp = parr_dc.data() + num0 * m_nchan * m_len_sft * (mbin - 1) + num1 * m_len_sft * (mbin - 1);
    //std::cout << std::endl;
    //for (int i = 0; i < m_len_sft; ++i)
    //{
    //    for (int ibin = 0; ibin < mbin-1; ++ibin)
    //    {
    //        std::cout << " pcmp[ " << i << "][" << ibin << "] =" << pcmp[i * (mbin - 1) + ibin];
    //    }
    //    std::cout << std::endl;
    //}
// !TEST of CHIRP CHANNEL
 
 


/*plan = fftwf_plan_many_dft(1, &nbin, nfft * nsub,
            in, &nbin,
            1, nbin,
            out, &nbin,
            1, nbin,
            FFTW_FORWARD, FFTW_MEASURE);*/
char str_wis_many[10000] = { 0 };
char str_wis_1[10000] = { 0 };

    // Define the size of the array
   //int ggg =  fftw_init_threads();
    //fftw_plan_with_nthreads(omp_get_max_threads());
    int rank = 1;
    const int n = 1<<20;
    int howmany = 6;


    int inembed = n;
    int istride = 1;
    int  idist = n;
    int onembed = n;
    int ostride = 1;
    int odist = n;

     // plan_many
    const int length = n * howmany;
    fftwf_complex* in = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * length);
    fftwf_complex* out = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * length);
    fftwf_plan plan_many = NULL;
    plan_many = fftwf_plan_many_dft(1, &n, howmany,
        in, &n,    1, n,       out, &n,    1, n,       FFTW_FORWARD, FFTW_MEASURE); 
   
    strcpy(str_wis_many, fftwf_export_wisdom_to_string());
    fftwf_destroy_plan(plan_many);

   
    // plan_1
    fftwf_plan plan_1 = NULL;
    plan_1 = fftwf_plan_dft_1d(n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    strcpy(str_wis_1, fftwf_export_wisdom_to_string());
    fftwf_destroy_plan(plan_1);

   //// fftwf_cleanup();
   // // Create the FFT plan
   // fftw_import_wisdom_from_string(str_wis_forward);
   // fftwf_plan plan = fftwf_plan_dft_1d(lengthChunk, pRawSignalCur, pffted_rowsignal, FFTW_FORWARD, FFTW_ESTIMATE);

   // // Execute the FFT
   // fftwf_execute(plan);
   // fftwf_destroy_plan(plan);

   /* fftwf_plan plan = NULL;
    plan = fftwf_plan_dft_1d(n, in, out, FFTW_FORWARD, FFTW_MEASURE);*/


    fftwf_free(in);
    fftwf_free(out);

    // Allocate memory for the array
    fftwf_complex* arinp = new fftwf_complex[length];
    fftwf_complex* arout = new fftwf_complex[length];

    // Fill the array with numbers from 1 to 12
    for (int i = 0; i < length; ++i) {
        arinp[i][0] = i + 1;
        arinp[i][1] = 0;
        arout[i][0] = 0.;
        arout[i][1] = 0.;
    }

    // Print the array on 4 lines, each line containing 4 numbers
    for (int i = 0; i < length; ++i)
    {
       /* std::cout << "(" << arinp[i][0] << "+" << arinp[i][1] << "i) ";
        if ((i + 1) % n == 0) {
            std::cout << std::endl;
        } */   
    }

    int NC = 100;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NC; ++i)
    {
        fftw_import_wisdom_from_string(str_wis_many);
        plan_many = fftwf_plan_many_dft(1, &n, howmany,
            arinp, &n, 1, n, arout, &n, 1, n, FFTW_FORWARD, FFTW_ESTIMATE);
        fftwf_execute(plan_many);
        fftwf_destroy_plan(plan_many);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time taken by function fncFdmtU_cu: " << duration.count() / ((double)NC) << " milliseconds" << std::endl;


    std::cout << std::endl;
    std::cout <<"OUTPUT PLAN_MANY"<< std::endl;
    // Print the array on 4 lines, each line containing 4 numbers
    for (int i = 0; i < length; ++i)
    {
        /*std::cout << "(" << arout[i][0] << "+" << arinp[i][1] << "i) ";
        if ((i + 1) % n == 0) {
            std::cout << std::endl;
        }*/
    }

    // plan_1
    // Fill the array with numbers from 1 to 12
    for (int i = 0; i < length; ++i) {
        arinp[i][0] = i + 1;
        arinp[i][1] = 0;
        arout[i][0] = 0.;
        arout[i][1] = 0.;
    }
    fftwf_cleanup();
    fftw_import_wisdom_from_string(str_wis_1);
    plan_1 = fftwf_plan_dft_1d(n, arinp, arout, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_execute(plan_1);   
    fftwf_destroy_plan(plan_1);
    std::cout << std::endl;
    std::cout << "OUTPUT PLAN_1" << std::endl;
    // Print the array on 4 lines, each line containing 4 numbers
    for (int i = 0; i < length; ++i)
    {
        /*std::cout << "(" << arout[i][0] << "+" << arinp[i][1] << "i) ";
        if ((i + 1) % n == 0) {
            std::cout << std::endl;
        }*/
    }
   // fftw_cleanup_threads();
    // Free the memory allocated for the array
    delete[] arinp;
    delete[] arout;
    return 0;    
}
