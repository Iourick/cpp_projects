#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Chunk_fly_gpu.cuh"
#include "yr_cart.h"
#include <vector>
#include "OutChunkHeader.h"

#include "Constants.h"


#include <chrono>

#include "helper_functions.h"
#include "helper_cuda.h"
#include <math_functions.h>
#include "aux_kernels.cuh"
#include "Detection.cuh"
#include "Cleaning.cuh"
#include <complex>

#include "Fragment.h"
#include "npy.hpp"
#include "TelescopeHeader.h"
extern cudaError_t cudaStatus0;


//  
//#ifdef _WIN32 // Windows
//
//#include <Windows.h>
//
//    void emitSound(int frequency, int duration) {
//        Beep(frequency, duration);
//    }
//
//#else // Linux
//
//#include <cmath>
//#include <alsa/asoundlib.h>
//
//    void emitSound(int frequency, int duration) {
//        int rate = 44100; // Sampling rate
//        snd_pcm_t* handle;
//        snd_pcm_open(&handle, "default", SND_PCM_STREAM_PLAYBACK, 0);
//        snd_pcm_set_params(handle, SND_PCM_FORMAT_S16_LE, SND_PCM_ACCESS_RW_INTERLEAVED, 1, rate, 1, 500000);
//
//        short buf[rate * duration];
//
//        for (int i = 0; i < rate * duration; i++) {
//            int sample = 32760 * sin(2 * M_PI * frequency * i / rate);
//            buf[i] = sample;
//        }
//
//        snd_pcm_writei(handle, buf, rate * duration);
//        snd_pcm_close(handle);
//    }
//
//#endif 
	

	extern const unsigned long long TOtal_GPU_Bytes;;

	
    #define BLOCK_DIM 32//16
	CChunk_fly_gpu::~CChunk_fly_gpu()
	{		
	}
	//-----------------------------------------------------------
	CChunk_fly_gpu::CChunk_fly_gpu() :CChunk_gpu()
	{			
	}
	//-----------------------------------------------------------

	CChunk_fly_gpu::CChunk_fly_gpu(const  CChunk_fly_gpu& R) :CChunk_gpu(R)
	{
		
	}
	//-------------------------------------------------------------------

	CChunk_fly_gpu& CChunk_fly_gpu::operator=(const CChunk_fly_gpu& R)
	{
		if (this == &R)
		{
			return *this;
		}
		CChunk_gpu:: operator= (R);
		
		return *this;
	}
	//------------------------------------------------------------------
	CChunk_fly_gpu::CChunk_fly_gpu(
		const float Fmin
		, const float Fmax
		, const int npol
		, const int nchan		
		, const unsigned int len_sft
		, const int Block_id
		, const int Chunk_id
		, const  float d_max
		, const  float d_min
		, const int ncoherent
		, const float sigma_bound
		, const int length_sum_wnd
		, const int nbin
		, const int nfft
		, const int noverlap
		, const float tsamp) : CChunk_gpu(Fmin
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
			, tsamp)
	{		
	}

//----------------------------------------------------------------------------------------
void CChunk_fly_gpu::compute_chirp_channel()
{	
}

//------------------------------------------------------------------------------------------
void  CChunk_fly_gpu::elementWiseMult(cufftComplex* d_arrOut, cufftComplex* d_arrInp0	, int  idm)
{
	cudaStatus0 = cudaGetLastError();
	if (cudaStatus0 != cudaSuccess) {
		fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus0));
		return;
	}
	double bw = m_Fmax - m_Fmin;
	int mbin = get_mbin();
	double bw_sub = bw / m_nchan;
	double bw_chan = bw_sub / m_len_sft;

	/*const dim3 block_Size2(256, 1, 1);
	const dim3 gridSize2((mbin + block_Size2.x - 1) / block_Size2.x, m_len_sft, m_nchan);
	kernel_el_wise_mult_onthe_fly_ << < gridSize2, block_Size2 >> >
		(d_arrOut, d_arrInp0, &m_pd_arrcoh_dm[idm], m_nchan, m_len_sft, mbin, m_nfft, m_npol, m_Fmin, bw_sub, bw_chan);*/

	const dim3 block_Size2(256, 1, 1);
	const dim3 gridSize2((mbin + block_Size2.x - 1) / block_Size2.x, m_len_sft, m_nchan);
	kernel_el_wise_mult_onthe_fly << < gridSize2, block_Size2 >> >
		(d_arrOut, d_arrInp0, &m_pd_arrcoh_dm[idm], m_nchan, m_len_sft, mbin, m_nfft, m_npol, m_Fmin, bw_sub, bw_chan);
	
	 cudaStatus0 = cudaGetLastError();
	 if (cudaStatus0 != cudaSuccess) {
		 fprintf(stderr, "cudaGetLastError failed: %s, idm = %i\n", cudaGetErrorString(cudaStatus0), idm);
		 return;
	 }
}


//-------------------------------------------------------------------------------------------
__global__
void kernel_el_wise_mult_onthe_fly(cufftComplex* parr_Out, cufftComplex* parr_Inp, double* pdm
	, int nchan, int len_sft, int mbin, int nfft, int npol, double Fmin, double bw_sub, double bw_chan)
{
	int ibin = blockIdx.x * blockDim.x + threadIdx.x;
	if (ibin >= mbin)
	{
		return;
	}
	int ichan = blockIdx.z;
	int isft = blockIdx.y;

	float temp0 = Fmin + bw_sub * (0.5 + ichan) + bw_chan * (isft - len_sft / 2.0 + 0.5);
	int i0 = ichan * len_sft * mbin + isft * mbin;

	//
	double temp = -0.5 * bw_chan + (ibin + 0.5) * bw_chan / mbin;
	double bin_freqs = temp;
	double taper = 1.0 / sqrt(1.0 + pow(temp / (0.47 * bw_chan), 80));

	double temp1 = bin_freqs / temp0;
	double phase_delay = ((*pdm) * temp1 * temp1 / (temp0 + bin_freqs) * 4.148808e9);
	double val_prD_int = 0;
	double t = -modf(phase_delay, &val_prD_int) * 2.0;
	double val_x = 0.0, val_y = 0.;

	sincospi(t, &val_y, &val_x);

	int nbin = mbin * len_sft;
	int num_el = i0 + ibin;
	int i1 = num_el / nbin;
	int ibin1 = num_el % nbin;
	ibin1 = (ibin1 + nbin / 2) % nbin;

	cufftComplex dc;
	//parr_dc[i1 * nbin + ibin1].x = float(val_x * parr_taper[ibin] / (double(nbin)));
	dc.x = float(val_x * taper / (double(nbin)));
	dc.y = float(val_y * taper / (double(nbin)));

	int ind = i1 * nbin + ibin1;
	int stride = nbin * nchan;
	for (int i = 0; i < nfft * npol / 2; ++i)
	{
		parr_Out[ind] = cmpMult(parr_Inp[ind], dc);
		ind += stride;
	}
}

//-------------------------------------------------------------------------------------------
__global__
void kernel_el_wise_mult_onthe_fly_(cufftComplex* parr_Out, cufftComplex* parr_Inp, double* pdm
	, int nchan, int len_sft, int mbin, int nfft, int npol, double Fmin, double bw_sub, double bw_chan)
{
	__shared__  char charr[sizeof(double) + sizeof(int)]; 
	

	int ibin = blockIdx.x * blockDim.x + threadIdx.x;
	if (ibin >= mbin)
	{
		return;
	}	

	int ichan = blockIdx.z;
	int isft = blockIdx.y;
	double* temp0 = (double*)charr;
	int* i0 = (int*)(temp0 + 1);
	temp0[0] = Fmin + bw_sub * (0.5 + ichan) + bw_chan * (isft - len_sft / 2.0 + 0.5);
	i0[0] = ichan * len_sft * mbin + isft * mbin;
	/*if ((ichan == 1) && (isft == 1)&&(blockIdx.x == 1))
	{
		printf("%f   %i", temp0[0], i0[0]);
	}*/
	double bin_freqs = -0.5 * bw_chan + (ibin + 0.5) * bw_chan / mbin;
	double taper = 1.0 / sqrt(1.0 + pow(bin_freqs / (0.47 * bw_chan), 80));

	double temp1 = bin_freqs / temp0[0];
	double phase_delay = ((*pdm) * temp1 * temp1 / (temp0[0] + bin_freqs) * 4.148808e9);
	double val_prD_int = 0;
	double t = -modf(phase_delay, &val_prD_int) * 2.0;
	double val_x = 0.0, val_y = 0.;

	sincospi(t, &val_y, &val_x);

	int nbin = mbin * len_sft;
	int num_el = i0[0] + ibin;
	int i1 = num_el / nbin;
	int ibin1 = num_el % nbin;
	ibin1 = (ibin1 + nbin / 2) % nbin;

	cufftComplex dc;	
	dc.x = float(val_x * taper / (double(nbin)));
	dc.y = float(val_y * taper / (double(nbin)));

	int ind = i1 * nbin + ibin1;
	int stride = nbin * nchan;
	for (int i = 0; i < nfft * npol / 2; ++i)
	{
		parr_Out[ind] = cmpMult(parr_Inp[ind], dc);
		ind += stride;
	}
}
//-----------------------------------------------------------------------
cufftComplex cmpMult(cufftComplex& a, cufftComplex&b)
{
	cufftComplex r;
	r.x = a.x * b.x - a.y * b.y;
	r.y = a.x * b.y + a.y * b.x;
	return r;
}

//---------------------------------------------------------------
//
//__global__ 
//void normalize_and_clean(fdmt_type_* parrOut, float* d_arr, const int NRows, const int NCols
//	,float *pmean, float *pstd, float* d_arrRowDisp, float *pmeanDisp, float *pstdDisp)
//{
//	__shared__ int sbad[1];
//	unsigned int i = threadIdx.x;
//	unsigned int irow = blockIdx.y;
//	if (i >= NCols)
//	{
//		return;
//	}
//	if (fabs(d_arrRowDisp[irow] - *pmeanDisp) > 4. * (*pstdDisp))
//	{
//		sbad[0] = 1;
//	}
//	else
//	{
//		sbad[0] = 0;
//	}
//	//--------------------------------
//	if (sbad[0] == 1)
//	{
//		while (i < NCols)
//		{
//			parrOut[irow * NCols + i] = 0;
//			i += blockDim.x;
//		}
//	}
//	else
//	{
//		while (i < NCols)
//		{
//			parrOut[irow * NCols + i] = (fdmt_type_)((d_arr[irow * NCols + i] - (*pmean) )/((*pstd )));
//			i += blockDim.x;
//		}
//	}
//	
//
//}


//-----------------------------------------------------------------------------------------
//void windowization(float* d_fdmt_normalized, const int Rows, const int Cols, const int width, float* parrImage)
//{
//	for (int i = 0; i < Rows; ++i)
//	{
//		for (int j = 0; j < Cols; ++j)
//		{
//			
//			float sum = 0.;
//			for (int k = 0; k < width; ++k)
//			{
//				if ((j + k) < Cols)
//				{
//					sum += d_fdmt_normalized[i * Cols + j + k];
//				}
//				else
//				{
//					sum = 0.;
//					break;
//				}
//				
//			}
//			parrImage[i * Cols + j] = sum / sqrt((float)width);
//		}
//	}
//}
////----------------------------------------------------
//__global__
//void fdmt_normalization(fdmt_type_* d_arr, fdmt_type_* d_norm, const int len, float* d_pOutArray)
//{
//
//	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
//	if (idx >= len)
//	{
//		return;
//	}
//	d_pOutArray[idx] = ((float)d_arr[idx]) / sqrtf(((float)d_norm[idx]) + 1.0E-8);
//
//}








