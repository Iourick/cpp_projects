#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Chunk_v1_gpu.cuh"
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

	// timing variables:
	  // fdmt time
	long long iFdmt_time1 = 0;
	// read && transform data time
	long long  iReadTransform_time1 = 0;
	// fft time
	long long  iFFT_time1 = 0;
	// detection time
	long long  iMeanDisp_time1= 0;
	// detection time
	long long  iNormalize_time1= 0;
	// total time
	long long  iTotal_time1= 0;
    #define BLOCK_DIM 32//16
	CChunk_v1_gpu::~CChunk_v1_gpu()
	{
		if (m_pd_arr_dc)
		{
			cudaFree(m_pd_arr_dc);
		}
	}
	//-----------------------------------------------------------
	CChunk_v1_gpu::CChunk_v1_gpu() :CChunk_gpu()
	{	
		m_pd_arr_dc = nullptr;
	}
	//-----------------------------------------------------------

	CChunk_v1_gpu::CChunk_v1_gpu(const  CChunk_v1_gpu& R) :CChunk_gpu(R)
	{
		cudaMalloc(&m_pd_arr_dc, R.m_coh_dm_Vector.size() * R.m_nchan * R.m_nbin * sizeof(cufftComplex));
		cudaMemcpy(m_pd_arr_dc, R.m_pd_arr_dc, R.m_coh_dm_Vector.size() * R.m_nchan * R.m_nbin * sizeof(cufftComplex), cudaMemcpyDeviceToDevice);

	}
	//-------------------------------------------------------------------

	CChunk_v1_gpu& CChunk_v1_gpu::operator=(const CChunk_v1_gpu& R)
	{
		if (this == &R)
		{
			return *this;
		}
		CChunk_gpu:: operator= (R);

		if (m_pd_arr_dc)
		{
			cudaFree(m_pd_arr_dc);
		}
		cudaMalloc(&m_pd_arr_dc, R.m_coh_dm_Vector.size() * R.m_nchan * R.m_nbin * sizeof(cufftComplex));
		cudaMemcpy(m_pd_arr_dc, R.m_pd_arr_dc, R.m_coh_dm_Vector.size() * R.m_nchan * R.m_nbin * sizeof(cufftComplex), cudaMemcpyDeviceToDevice);
		return *this;
	}
	//------------------------------------------------------------------
	CChunk_v1_gpu::CChunk_v1_gpu(
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
		cudaMalloc(&m_pd_arr_dc, m_coh_dm_Vector.size() * m_nchan * m_nbin * sizeof(cufftComplex));
		cudaStatus0 = cudaGetLastError();
		if (cudaStatus0 != cudaSuccess) {
			fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus0));
			return;
		}
		compute_chirp_channel();
		cudaStatus0 = cudaGetLastError();
		if (cudaStatus0 != cudaSuccess) {
			fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus0));
			return;
		}
	}


void CChunk_v1_gpu::compute_chirp_channel()
{
	// 1 preparations
	 double bw = m_Fmax - m_Fmin;
	int mbin = get_mbin();
	 double bw_sub = bw / m_nchan;
	 double bw_chan = bw_sub / m_len_sft;
	int ndm = m_coh_dm_Vector.size();
	//1!

	
	double* d_parr_freqs_chan = nullptr;
	cudaMalloc(&d_parr_freqs_chan, m_nchan * m_len_sft * sizeof(double));

	const dim3 block_size(32, 1, 1);
	const dim3 gridSize((m_len_sft + block_size.x - 1) / block_size.x, m_nchan, 1);
	kernel_create_arr_freqs_chan << < gridSize, block_size >> >(d_parr_freqs_chan, m_len_sft, bw_chan, m_Fmin, bw_sub);
	// 3!
	cudaStatus0 = cudaGetLastError();
	if (cudaStatus0 != cudaSuccess) {
		fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus0));
		return;
	}
	//int lenarr1 = m_nchan * m_len_sft;// *sizeof(cufftComplex));
	//std::vector<double> data1(lenarr1, 0);
	//cudaMemcpy(data1.data(), d_parr_freqs_chan, lenarr1 * sizeof(double), cudaMemcpyDeviceToHost);
	//cudaDeviceSynchronize();
	//std::array<long unsigned, 1> leshape128{ lenarr1 };
	//npy::SaveArrayAsNumpy("d_parr_freqs_chan.npy", false, leshape128.size(), leshape128.data(), data1);
	

	double* d_parr_taper = nullptr;
	cudaMalloc(&d_parr_taper, mbin * sizeof(double));
	double* d_parr_bin_freqs = nullptr;
	cudaMalloc(&d_parr_bin_freqs, mbin * sizeof(double));

	const dim3 block_Size1(1024, 1, 1);
	const dim3 gridSize1((mbin + block_Size1.x - 1) / block_Size1.x, 1, 1);
	kernel_create_arr_bin_freqs_and_taper << < gridSize1, block_Size1 >> > (d_parr_bin_freqs, d_parr_taper, bw_chan, mbin);

	cudaStatus0 = cudaGetLastError();
	if (cudaStatus0 != cudaSuccess) {
		fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus0));
		return;
	}
	//int lenarr3 = mbin;// *sizeof(cufftComplex));
	//std::vector<double> data3(lenarr3, 0);
	//cudaMemcpy(data3.data(), d_parr_bin_freqs, lenarr3 * sizeof(double), cudaMemcpyDeviceToHost);
	//cudaDeviceSynchronize();
	//std::array<long unsigned, 1> leshape1{ lenarr3 };
	//npy::SaveArrayAsNumpy("parr_bin_freqs.npy", false, leshape1.size(), leshape1.data(), data3);

	


	/*int deviceId;
	cudaGetDevice(&deviceId);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, deviceId);
	std::cout << "Maximum number of blocks per grid: " << deviceProp.maxGridSize[0] << std::endl;
	std::cout << "Maximum number of threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;*/
	auto start = std::chrono::high_resolution_clock::now();

	

	const dim3 block_Size2(256,2,1);
	const dim3 gridSize2((mbin + block_Size2.x - 1) / block_Size2.x, (m_len_sft * m_nchan + block_Size2.y - 1) / block_Size2.y, (ndm + block_Size2.z -1)/ block_Size2.z);
	kernel_create_arr_dc << < gridSize2, block_Size2 >> > (m_pd_arr_dc, m_pd_arrcoh_dm, d_parr_freqs_chan, d_parr_bin_freqs, d_parr_taper, ndm, m_nchan, m_len_sft, mbin);
	cudaStatus0 = cudaGetLastError();
	if (cudaStatus0 != cudaSuccess) {
		fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus0));
		return;
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	std::cout << "Time taken by function fncFdmtU_cu: " << duration.count() << " microseconds" << std::endl;
	cudaFree(d_parr_freqs_chan);
	cudaFree(d_parr_taper);
	cudaFree(d_parr_bin_freqs);	
	cudaStatus0 = cudaGetLastError();
	if (cudaStatus0 != cudaSuccess) {
		fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus0));
		return;
	}
}
//-------------------------------------------------------------------------------------------
__global__
void kernel_create_arr_dc  (cufftComplex* parr_dc, double * parrcoh_dm,double* parr_freqs_chan, double* parr_bin_freqs, double *parr_taper
	, int ndm, int nchan, int len_sft, int mbin)
{
	//__shared__  double temp0[1];
	//__shared__  int i0[1];
	
	int ibin = blockIdx.x * blockDim.x + threadIdx.x;
	if (ibin >= mbin)
	{
		return;
	}	
	int num1 = blockIdx.y * blockDim.y + threadIdx.y;
	if (num1 >= nchan * len_sft)
	{
		return;
	}
	
	int ichan = num1 / len_sft;	
	int isft = num1 % len_sft;

	int idm = blockIdx.z * blockDim.z + threadIdx.z;
	if (idm >= ndm)
	{
		return;
	}

	float temp0 = parr_freqs_chan[ichan * len_sft + isft];
	int i0 = idm * nchan * len_sft * mbin + ichan * len_sft * mbin + isft * mbin;

	double temp1 = parr_bin_freqs[ibin] / temp0;
	double phase_delay=(parrcoh_dm[idm] * temp1 * temp1 / (temp0 + parr_bin_freqs[ibin]) * 4.148808e9);
	double val_prD_int = 0;
	double t = -modf(phase_delay, &val_prD_int) * 2.0;
	double val_x = 0.0, val_y = 0.;	
	
	sincospi(t, &val_y, &val_x);

	int nbin = mbin * len_sft;
	int num_el = i0 + ibin;
	int i1 = num_el / nbin;
	int ibin1 = num_el %nbin;
	ibin1 = (ibin1 + nbin / 2) % nbin;
	
	
	parr_dc[i1 * nbin + ibin1].x = float(val_x * parr_taper[ibin] / (double(nbin)));
	
	parr_dc[i1 * nbin + ibin1].y = float(val_y * parr_taper[ibin] / (double(nbin)));
}	
//-----------------------------------------------------------------------------
__global__ 	void  transpose_unpadd_intensity(float* fbuf, float* arin, int nfft, int noverlap_per_channel
	, int mbin_adjusted, const int nsub, const int nchan, int mbin)
{
	int  ibin = blockIdx.x * blockDim.x + threadIdx.x;
	if (!(ibin < mbin_adjusted))
	{
		return;
	}

	int ifft = blockIdx.z;
	int isub = blockIdx.y / nchan;
	int ichan = blockIdx.y % nchan;
	int ibin_adjusted = ibin + noverlap_per_channel;
	int isamp = ibin + mbin_adjusted * ifft;
	int msamp = mbin_adjusted * nfft;

	// Select bins from valid region and reverse the frequency axis		
	// printf("ipol = %i   ifft =  %i\n", ipol, ifft);
	int iinp = ifft * nsub * nchan * mbin + (nsub - isub - 1) * nchan * mbin + ichan * mbin + ibin_adjusted;
	int iout = isamp * nsub * nchan + isub * nchan + nchan - ichan - 1;
	// Select bins from valid region and reverse the frequency axis		

	if ((ifft == 0) && (isub == 0) && (ichan == 0) && (ibin < 10))
	{
		//printf("ibin = %i  arin[iinp] = %f\n", ibin, arin[iinp]);
	}
	fbuf[iout] = arin[iinp];

}
//--------------------------------------------------------------------------------------


void  CChunk_v1_gpu::elementWiseMult(cufftComplex* d_arrOut, cufftComplex* d_arrInp0	, int  idm)
{
	dim3 threadsPerBlock(1024, 1, 1);
	dim3 blocksPerGrid((m_nchan * m_nbin + threadsPerBlock.x - 1) / threadsPerBlock.x, m_nfft, m_npol / 2);
	element_wise_cufftComplex_mult_kernel << < blocksPerGrid, threadsPerBlock >> >
		(d_arrOut, d_arrInp0, &m_pd_arr_dc[m_nchan * m_nbin * idm], m_npol / 2, m_nfft, m_nchan * m_nbin);
	cudaDeviceSynchronize();
	// result stored in m_pdcmpbuff_ewmulted
	cudaStatus0 = cudaGetLastError();
	if (cudaStatus0 != cudaSuccess) {
		fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus0));
		return;
	}
}

//----------------------------------------------------------------------------------------
// #define BLOCK_DIM 32//16
//dim3 grid((width + BLOCK_DIM - 1) / BLOCK_DIM, (height + BLOCK_DIM - 1) / BLOCK_DIM, 1);
//dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);
//https://github.com/JonathanWatkins/CUDA/blob/master/NvidiaCourse/Exercises/transpose/transpose.cu
__global__ void transpose_(float* odata, float* idata, int width, int height)
{
	__shared__ float block[BLOCK_DIM][BLOCK_DIM + 1];

	// read the matrix tile into shared memory
		// load one element per thread from device memory (idata) and store it
		// in transposed order in block[][]
	unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
	unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
	if ((xIndex < width) && (yIndex < height))
	{
		unsigned int index_in = yIndex * width + xIndex;
		block[threadIdx.y][threadIdx.x] = idata[index_in];
	}

	// synchronise to ensure all writes to block[][] have completed
	__syncthreads();

	// write the transposed matrix tile to global memory (odata) in linear order
	xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
	yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
	if ((xIndex < height) && (yIndex < width))
	{
		unsigned int index_out = yIndex * height + xIndex;
		odata[index_out] = block[threadIdx.x][threadIdx.y];
	}
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








