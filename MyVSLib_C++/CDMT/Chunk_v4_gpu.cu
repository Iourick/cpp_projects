#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Chunk_v4_gpu.cuh"
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

    #define BLOCK_DIM 32//16
	CChunk_v4_gpu::~CChunk_v4_gpu()
	{
		if (m_pd_buff_dc)
		{
			cudaFree(m_pd_buff_dc);
		}

		if (m_pdarr_taper)
		{
			cudaFree(m_pdarr_taper);
		}
		if (m_pdarr_bin_freqs)
		{
			cudaFree(m_pdarr_bin_freqs);
		}
		
	}
	//-----------------------------------------------------------
	CChunk_v4_gpu::CChunk_v4_gpu() :CChunk_gpu()
	{	
		m_pd_buff_dc = nullptr;
		m_pdarr_bin_freqs = nullptr;
		m_pdarr_taper = nullptr;
	}
	//-----------------------------------------------------------

	CChunk_v4_gpu::CChunk_v4_gpu(const  CChunk_v4_gpu& R) :CChunk_gpu(R)
	{
		cudaMalloc(&m_pd_buff_dc, R.m_nchan * R.m_nbin * sizeof(cufftComplex));
        
		int mbin = get_mbin();
		cudaMalloc(&m_pdarr_bin_freqs, mbin * sizeof(double));
		cudaMemcpy(m_pdarr_bin_freqs, R.m_pdarr_bin_freqs, mbin * sizeof(double), cudaMemcpyDeviceToDevice);

		cudaMalloc(&m_pdarr_taper, mbin * sizeof(double));
		cudaMemcpy(m_pdarr_taper, R.m_pdarr_taper, mbin * sizeof(double), cudaMemcpyDeviceToDevice);

	}
	//-------------------------------------------------------------------

	CChunk_v4_gpu& CChunk_v4_gpu::operator=(const CChunk_v4_gpu& R)
	{
		if (this == &R)
		{
			return *this;
		}
		CChunk_gpu:: operator= (R);

		if (m_pd_buff_dc)
		{
			cudaFree(m_pd_buff_dc);
		}
		cudaMalloc(&m_pd_buff_dc,  R.m_nchan * R.m_nbin * sizeof(cufftComplex));


		int mbin = get_mbin();
		if (m_pdarr_bin_freqs)
		{
			cudaFree(m_pdarr_bin_freqs);
		}
		cudaMalloc(&m_pdarr_bin_freqs, mbin * sizeof(double));
		cudaMemcpy(m_pdarr_bin_freqs, R.m_pdarr_bin_freqs, mbin * sizeof(double), cudaMemcpyDeviceToDevice);

		if (m_pdarr_taper)
		{
			cudaFree(m_pdarr_taper);
		}
		cudaMalloc(&m_pdarr_taper, mbin * sizeof(double));
		cudaMemcpy(m_pdarr_taper, R.m_pdarr_taper, mbin * sizeof(double), cudaMemcpyDeviceToDevice);

		return *this;
	}
	//------------------------------------------------------------------
	CChunk_v4_gpu::CChunk_v4_gpu(
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
		cudaMalloc(&m_pd_buff_dc,  m_nchan * m_nbin * sizeof(cufftComplex));
		cudaStatus0 = cudaGetLastError();
		if (cudaStatus0 != cudaSuccess) {
			fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus0));
			return;
		}

		int mbin = get_mbin();
		
		cudaMalloc(&m_pdarr_bin_freqs, mbin * sizeof(double));		
		
		cudaMalloc(&m_pdarr_taper, mbin * sizeof(double));
		
		double bw = m_Fmax - m_Fmin;		
		double bw_sub = bw / m_nchan;
		double bw_chan = bw_sub / m_len_sft;
		const dim3 block_Size1(1024, 1, 1);
		const dim3 gridSize1((mbin + block_Size1.x - 1) / block_Size1.x, 1, 1);
		kernel_create_arr_bin_freqs_and_taper << < gridSize1, block_Size1 >> > (m_pdarr_bin_freqs, m_pdarr_taper, bw_chan, mbin);

	}


	void CChunk_v4_gpu::compute_chirp_channel()
	{		
	}

	void CChunk_v4_gpu::compute_current_chirp(int idm)
	{
		// 1 preparations
		 double bw = m_Fmax - m_Fmin;
		int mbin = get_mbin();
		 double bw_sub = bw / m_nchan;
		 double bw_chan = bw_sub / m_len_sft;
	
		//1!		

		auto start = std::chrono::high_resolution_clock::now();
		
		
		const dim3 block_Size2(256,2,1);
		const dim3 gridSize2((mbin + block_Size2.x - 1) / block_Size2.x, (m_len_sft * m_nchan + block_Size2.y - 1) / block_Size2.y,1);
		kernel_create_current_chirp_ << < gridSize2, block_Size2 >> > (m_pd_buff_dc, &m_pd_arrcoh_dm[idm],  m_pdarr_bin_freqs, m_pdarr_taper
			,  m_nchan, m_len_sft, mbin, (double)m_Fmin, bw_sub,  bw_chan);
		cudaStatus0 = cudaGetLastError();
		if (cudaStatus0 != cudaSuccess) {
			fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus0));
			return;
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
		//std::cout << "Time taken by function fncFdmtU_cu: " << duration.count() << " microseconds" << std::endl;		
		
		cudaStatus0 = cudaGetLastError();
		if (cudaStatus0 != cudaSuccess) {
			fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus0));
			return;
		}
	}
//-------------------------------------------------------------------------------------------
__global__
void kernel_create_current_chirp_(cufftComplex* parr_dc, double * pdm, double* parr_bin_freqs, double *parr_taper
	, int nchan, int len_sft, int mbin, double Fmin, double bw_sub, double bw_chan)
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
	
	float temp0 = Fmin + bw_sub * (0.5 + ichan) + bw_chan * (isft - len_sft / 2.0 + 0.5);

	int i0 = ichan * len_sft * mbin + isft * mbin;

	double temp1 = parr_bin_freqs[ibin] / temp0;
	double phase_delay = ((*pdm) * temp1 * temp1 / (temp0 + parr_bin_freqs[ibin]) * 4.148808e9);
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



void  CChunk_v4_gpu::elementWiseMult(cufftComplex* d_arrOut, cufftComplex* d_arrInp0	, int  idm)
{
	compute_current_chirp(idm);

	dim3 threadsPerBlock(1024, 1, 1);
	dim3 blocksPerGrid((m_nchan * m_nbin + threadsPerBlock.x - 1) / threadsPerBlock.x, m_nfft, m_npol / 2);
	element_wise_cufftComplex_mult_kernel << < blocksPerGrid, threadsPerBlock >> >
		(d_arrOut, d_arrInp0, m_pd_buff_dc, m_npol / 2, m_nfft, m_nchan * m_nbin);
	cudaDeviceSynchronize();
	// result stored in m_pdcmpbuff_ewmulted
	cudaStatus0 = cudaGetLastError();
	if (cudaStatus0 != cudaSuccess) {
		fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus0));
		return;
	}
}

//----------------------------------------------------------------------------------------








