
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

inline int get_noverlap_per_channel(int  noverlap , int nchan)
{
    return noverlap / nchan;
};

inline int get_mbin(int nbin , int nchan)
{
    return  nbin / nchan;
}

inline int get_mbin_adjusted(int nbin, int nchan, int  noverlap)
{
    return get_mbin(nbin, nchan) - 2 * get_noverlap_per_channel(noverlap, nchan);
}



inline int get_msamp(int nfft, int nbin, int nchan, int  noverlap)
{
    return nfft * get_mbin_adjusted( nbin, nchan,  noverlap);
}


    //--------------------------------------------------------------------------------
//	
// 	nsub = nchan
// nchan = len_sft
__global__	void  transpose_unpadd_gpu(int* fbuf, int* arin, int nfft, int noverlap_per_channel
    , int mbin_adjusted, const int nsub, const int nchan, int mbin)
{
    int  ibin = blockIdx.x * blockDim.x + threadIdx.x;
    if (!(ibin < mbin_adjusted))
    {
        return;
    }
    int ipol = blockIdx.z / nfft;
    int ifft = blockIdx.z % nfft;
    int isub = blockIdx.y / nchan;
    int ichan = blockIdx.y % nchan;
    int ibin_adjusted = ibin + noverlap_per_channel;
    int isamp = ibin + mbin_adjusted * ifft;
    int msamp = mbin_adjusted * nfft;
    // Select bins from valid region and reverse the frequency axis		
   // printf("ipol = %i   ifft =  %i\n", ipol, ifft);
    int iinp = ipol *nfft * nsub * nchan * mbin +       ifft * nsub * nchan * mbin + (nsub - isub - 1) * nchan * mbin + ichan * mbin + ibin_adjusted;
    int iout = ipol * msamp  *nsub * nchan +      isamp * nsub * nchan + isub * nchan + nchan - ichan - 1;
    fbuf[iout] =        arin[iinp];
    printf(" ifft =  %i  isub = %i  ichan = %i ibin = %i   iinp = %i  iout = %i\n",ifft, isub, ichan, ibin, iinp, iout);
   
}
void transpose_unpadd(int* fbuf, int* arin, int nfft, int noverlap_per_channel, int mbin_adjusted, const int nsub, const int nchan, int mbin)
{
    for (int ifft = 0; ifft < nfft; ++ifft)
    {
        for (int isub = 0; isub < nsub; ++isub)
        {
            for (int ichan = 0; ichan < nchan; ++ichan)
            {
                for (int ibin = 0; ibin < mbin_adjusted; ++ibin)
                {
                    int ibin_adjusted = ibin + noverlap_per_channel;
                    int isamp = ibin + mbin_adjusted * ifft;

                    // Select bins from valid region and reverse the frequency axis
                    fbuf[isamp * nsub * nchan + isub * nchan + nchan - ichan - 1] =
                        arin[ifft * nsub * nchan * mbin + (nsub - isub - 1) * nchan * mbin + ichan * mbin + ibin_adjusted];

                    int iinp = ifft * nsub * nchan * mbin + (nsub - isub - 1) * nchan * mbin + ichan * mbin + ibin_adjusted;
                    int iout = isamp * nsub * nchan + isub * nchan + nchan - ichan - 1;
                    printf(" ifft =  %i  isub = %i  ichan = %i ibin = %i   iinp = %i  iout = %i\n", ifft, isub, ichan, ibin, iinp, iout);
                }
            }
        }
    }
}

int main()
{
    const int nfft = 2;
    const int noverlap= 2;
    int nbin = 8;
    int nsub = 2;
    int nchan = 2;
    int npol = 4;

    const int  noverlap_per_channel =   get_noverlap_per_channel(  noverlap,nchan);
    const int mbin_adjusted = get_mbin_adjusted( nbin,  nchan,  noverlap);
    const int mbin = get_mbin(nbin, nchan);
    const int msamp = get_msamp( nfft, nbin,nchan,   noverlap);


    int *arin = new int[nfft * nsub * nchan * mbin];
    int *fbuf =  new int [nfft * nsub * nchan * mbin];

    // Fill arin with some sample data
    for (int i = 0; i < nfft * nsub * nchan * mbin; ++i)
    {
        arin[i] = i + 1; // Some arbitrary values
    }

    // Print the input array
    std::cout << "Input array (arin):" << std::endl;
    for (int i = 0; i < nfft * nsub * nchan * mbin; ++i)
    {
        std::cout << arin[i] << " ";
    }
    std::cout << std::endl;

    // Call the transpose_unpadd function
    transpose_unpadd(fbuf, arin, nfft, noverlap_per_channel, mbin_adjusted, nsub, nchan, mbin);

    // Print the output array
    std::cout << "Output array (fbuf):" << std::endl;
    for (int i = 0; i < msamp; ++i)
    {
        std::cout << std::endl;
        for (int j = 0; j < nsub * nchan; ++j)
        {
            std::cout << fbuf[i * nsub * nchan + j] << " ; ";
        }
    }
    std::cout << std::endl;

    // Now, you can write assertions to verify if the output is as expected
    // For simplicity, let's just check if some values match

    // Check a specific value in the output buffer
    if (fbuf[0] == arin[mbin_adjusted * nsub * nchan + nchan * mbin + 1])
    {
        std::cout << "Test passed!" << std::endl;
    }
    else
    {
        std::cout << "Test failed!" << std::endl;
    }
//----------------------------------------------------------------------------------
    int* darrin = NULL;
    cudaMalloc((void**)&darrin, nfft * nsub * nchan * mbin * sizeof(int));
    cudaMemcpy(darrin, arin, nfft * nsub * nchan * mbin * sizeof(int), cudaMemcpyHostToDevice);

    int* dfbuf = NULL;
    cudaMalloc((void**)&dfbuf, msamp* nsub * nchan * sizeof(int));


    dim3 treads(4, 1, 1);
    dim3 blocks((mbin_adjusted + treads.x - 1) / treads.x, nsub * nchan,  nfft);
    transpose_unpadd_gpu << < blocks, treads >> > (dfbuf, darrin, nfft, noverlap_per_channel
        , mbin_adjusted, nsub,  nchan,mbin);
    // Add more assertions as needed to thoroughly test your function

    cudaMemcpy(fbuf, dfbuf, msamp * nsub * nchan * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < msamp; ++i)
    {
        std::cout << std::endl;
        for (int j = 0; j < nsub * nchan; ++j)
        {
            std::cout << fbuf[i * nsub * nchan + j] << " ; ";
        }
    }
    delete[]arin;
    delete[]fbuf;
    cudaFree(dfbuf);
    cudaFree(darrin);
    return 0;
}

