#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Chunk_py_gpu.cuh"
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
	CChunk_py_gpu::~CChunk_py_gpu()
	{
		if (m_pd_arr_dc_py)
		{
			cudaFree(m_pd_arr_dc_py);
		}
	}
	//-----------------------------------------------------------
	CChunk_py_gpu::CChunk_py_gpu() :CChunk_gpu()
	{		
		m_pd_arr_dc_py = nullptr;
	}
	//-----------------------------------------------------------

	CChunk_py_gpu::CChunk_py_gpu(const  CChunk_py_gpu& R) :CChunk_gpu(R)
	{
		cudaMalloc(&m_pd_arr_dc_py, R.m_coh_dm_Vector.size() * R.m_nchan * R.m_nbin * sizeof(cufftComplex));
		cudaMemcpy(m_pd_arr_dc_py, R.m_pd_arr_dc_py, R.m_coh_dm_Vector.size() * R.m_nchan * R.m_nbin * sizeof(cufftComplex), cudaMemcpyDeviceToDevice);
	}
	//-------------------------------------------------------------------

	CChunk_py_gpu& CChunk_py_gpu::operator=(const CChunk_py_gpu& R)
	{
		if (this == &R)
		{
			return *this;
		}
		CChunk_gpu:: operator= (R);
		if (m_pd_arr_dc_py)
		{
			cudaFree(m_pd_arr_dc_py);
		}
		cudaMalloc(&m_pd_arr_dc_py, R.m_coh_dm_Vector.size() * R.m_nchan * R.m_nbin * sizeof(cufftComplex));
		cudaMemcpy(m_pd_arr_dc_py, R.m_pd_arr_dc_py, R.m_coh_dm_Vector.size() * R.m_nchan * R.m_nbin * sizeof(cufftComplex), cudaMemcpyDeviceToDevice);


		return *this;
	}
	//------------------------------------------------------------------
	CChunk_py_gpu::CChunk_py_gpu(
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
		cudaMalloc(&m_pd_arr_dc_py, m_coh_dm_Vector.size() * m_nchan * m_nbin * sizeof(cufftComplex));
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


void CChunk_py_gpu::compute_chirp_channel()
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

	//int lenarr3 = mbin;// *sizeof(cufftComplex));
	//std::vector<double> data3(lenarr3, 0);
	//cudaMemcpy(data3.data(), d_parr_bin_freqs, lenarr3 * sizeof(double), cudaMemcpyDeviceToHost);
	//cudaDeviceSynchronize();
	//std::array<long unsigned, 1> leshape1{ lenarr3 };
	//npy::SaveArrayAsNumpy("parr_bin_freqs.npy", false, leshape1.size(), leshape1.data(), data3);

	//int lenarr4 = mbin;// *sizeof(cufftComplex));
	//std::vector<double> data4(lenarr4, 0);
	//cudaMemcpy(data4.data(), d_parr_taper, lenarr4 * sizeof(std::complex<float>), cudaMemcpyDeviceToHost);
	//cudaDeviceSynchronize();
	//std::array<long unsigned, 1> leshape2{ lenarr4 };
	//npy::SaveArrayAsNumpy("parr_taper.npy", false, leshape2.size(), leshape2.data(), data4);


	/*int deviceId;
	cudaGetDevice(&deviceId);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, deviceId);
	std::cout << "Maximum number of blocks per grid: " << deviceProp.maxGridSize[0] << std::endl;
	std::cout << "Maximum number of threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;*/
	auto start = std::chrono::high_resolution_clock::now();

	

	const dim3 block_Size2(512,2,1);
	const dim3 gridSize2((mbin + block_Size2.x - 1) / block_Size2.x, (m_len_sft * m_nchan + block_Size2.y - 1) / block_Size2.y, (ndm + block_Size2.z -1)/ block_Size2.z);
	kernel_create_arr_dc_py << < gridSize2, block_Size2 >> > (m_pd_arr_dc_py, m_pd_arrcoh_dm, d_parr_freqs_chan, d_parr_bin_freqs, d_parr_taper, ndm, m_nchan, m_len_sft, mbin);
	
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	std::cout << "Time taken by function fncFdmtU_cu: " << duration.count() << " microseconds" << std::endl;
	cudaFree(d_parr_freqs_chan);
	cudaFree(d_parr_taper);
	cudaFree(d_parr_bin_freqs);


	

	//int lenarr4 = ndm* m_nchan* m_len_sft* mbin;// *sizeof(cufftComplex));
	//std::vector<complex<float>> data4(lenarr4, 0);
	//cudaMemcpy(data4.data(), m_pd_arr_dc_py, lenarr4 * sizeof(std::complex<float>), cudaMemcpyDeviceToHost);
	//cudaDeviceSynchronize();
	//std::array<long unsigned, 1> leshape2{ lenarr4 };
	//npy::SaveArrayAsNumpy("arr_dc.npy", false, leshape2.size(), leshape2.data(), data4);
}


//-------------------------------------------------------------------------------------------
__global__
void kernel_create_arr_dc_py(cufftComplex* parr_dc, double* parrcoh_dm, double* parr_freqs_chan, double* parr_bin_freqs, double* parr_taper
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
	double phase_delay = (parrcoh_dm[idm] * temp1 * temp1 / (temp0 + parr_bin_freqs[ibin]) * 4.148808e9);
	double val_prD_int = 0;
	double t = -modf(phase_delay, &val_prD_int) * 2.0;
	double val_x = 0.0, val_y = 0.;

	sincospi(t, &val_y, &val_x);
	parr_dc[i0 + ibin].x = float(val_x * parr_taper[ibin]);
	parr_dc[i0 + ibin].y = float(val_y * parr_taper[ibin]);
}

//---------------------------------------------------
bool CChunk_py_gpu::process(void* pcmparrRawSignalCur
	, std::vector<COutChunkHeader>* pvctSuccessHeaders, std::vector<std::vector<float>>* pvecImg)
{
	// 1. 
	const int mbin = get_mbin();
	const int noverlap_per_channel = get_noverlap_per_channel();
	const int mbin_adjusted = get_mbin_adjusted();
	const int msamp = get_msamp();
	const int mchan = m_nchan * m_len_sft;
	// 1!
	//int lenarr4 = m_nfft * m_nchan * m_nbin * (m_npol / 2) / 2;// *sizeof(cufftComplex));
	//std::vector<complex<float>> data4(lenarr4, 0);
	//cudaMemcpy(data4.data(), (cufftComplex*)pcmparrRawSignalCur, lenarr4 * sizeof(std::complex<float>), cudaMemcpyDeviceToHost);
	//cudaDeviceSynchronize();
	//std::array<long unsigned, 1> leshape2{ lenarr4 };
	//npy::SaveArrayAsNumpy("pcmparrRawSignalFfted.npy", false, leshape2.size(), leshape2.data(), data4);
	//int ii = 0;
	// 2. Forward FFT execution
	checkCudaErrors(cufftExecC2C(m_fftPlanForward, (cufftComplex*)pcmparrRawSignalCur, m_pdcmpbuff_ewmulted, CUFFT_FORWARD));
	//2!	
	
	

	//3. roll and normalize ffted signals
	dim3 treads_per_block(256, 1);
	dim3 blocks_per_grid((m_nbin + treads_per_block.x - 1) / treads_per_block.x, m_nfft * m_nchan * m_npol / 2);

	//cufftComplex* pcmparrRawSignalRolled = NULL;
	//cudaMalloc(&pcmparrRawSignalRolled, m_nfft * m_nchan * m_npol / 2 * m_nbin * sizeof(cufftComplex));
	dim3 threads(1024, 1);
	dim3 blocks((m_nbin + threads.x - 1) / threads.x, m_nfft * m_nchan * m_npol / 2);
	roll_rows_and_normalize_kernel << < blocks, threads >> > ((cufftComplex*)pcmparrRawSignalCur, m_pdcmpbuff_ewmulted, m_nfft * m_nchan * m_npol / 2, m_nbin, m_nbin / 2);

	cudaStatus0 = cudaGetLastError();
	if (cudaStatus0 != cudaSuccess) {
		fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus0));
		return false;
	}

	
	cudaStatus0 = cudaGetLastError();
	if (cudaStatus0 != cudaSuccess) {
		fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus0));
		return false;
	}
	//int lenarr4 = m_nfft * m_nchan * m_nbin * (m_npol / 2) / 2;// *sizeof(cufftComplex));
	//std::vector<complex<float>> data4(lenarr4, 0);
	//cufftComplex* pc = &pcmparrRawSignalRolled[lenarr4];
	//cudaMemcpy(data4.data(), pcmparrRawSignalRolled, lenarr4 * sizeof(std::complex<float>), cudaMemcpyDeviceToHost);
	//cudaDeviceSynchronize();
	//std::array<long unsigned, 1> leshape2{ lenarr4 };
	//npy::SaveArrayAsNumpy("pcmparrRawSignalFfted.npy", false, leshape2.size(), leshape2.data(), data4);
	//int ii = 0;

	//3!

	//float* poutImg = NULL;
	//cudaMalloc(&poutImg, msamp * m_len_sft * m_coh_dm_Vector.size() * sizeof(float));
	for (int idm = 0; idm < m_coh_dm_Vector.size(); ++idm)
	{
		dim3 threadsPerBlock(1024, 1, 1);
		dim3 blocksPerGrid((m_nchan * m_nbin + threadsPerBlock.x - 1) / threadsPerBlock.x, m_nfft, m_npol / 2);
		element_wise_cufftComplex_mult_kernel << < blocksPerGrid, threadsPerBlock >> >
			(m_pdcmpbuff_ewmulted,(cufftComplex*)pcmparrRawSignalCur, &m_pd_arr_dc_py[m_nchan * m_nbin * idm], m_npol / 2, m_nfft, m_nchan * m_nbin);
		cudaDeviceSynchronize();
		// result stored in m_pdcmpbuff_ewmulted
		cudaStatus0 = cudaGetLastError();
		if (cudaStatus0 != cudaSuccess) {
			fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus0));
			return false;
		}

		//int lenarr4 = m_nfft * m_nchan * m_nbin * (m_npol / 2) / 2;// *sizeof(cufftComplex));
		//std::vector<complex<float>> data4(lenarr4, 0);
		//cufftComplex* pc = &pcmparrRawSignalRolled[lenarr4];
		//cudaMemcpy(data4.data(), (cufftComplex*)pcmparrRawSignalCur, lenarr4 * sizeof(std::complex<float>), cudaMemcpyDeviceToHost);
		//cudaDeviceSynchronize();
		//std::array<long unsigned, 1> leshape2{ lenarr4 };
		//npy::SaveArrayAsNumpy("pcmparrRawSignalFfted.npy", false, leshape2.size(), leshape2.data(), data4);
		//int ii = 0;

		int threads1 = 1024;
		int blocks1 = (m_npol / 2 * m_nfft * m_nchan * m_nbin + threads1 - 1) / threads1;
		/*divide_cufftComplex_array_kernel << <blocks1, threads1 >> > ((cufftComplex*)pcmparrRawSignalCur, m_npol / 2 * m_nfft * m_nchan * m_nbin, ((float)m_nbin));
		cudaDeviceSynchronize();*/

		cudaStatus0 = cudaGetLastError();
		if (cudaStatus0 != cudaSuccess) {
			fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus0));
			return false;
		}
		/*	int lenarr4 = m_nfft * m_nchan * m_nbin * (m_npol / 2) / 2;
		std::vector<complex<float>> data4(lenarr4, 0);
		cudaMemcpy(data4.data(), &((cufftComplex*)pcmparrRawSignalCur)[lenarr4], lenarr4 * sizeof(std::complex<float>), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		std::array<long unsigned, 1> leshape2{ lenarr4 };
		npy::SaveArrayAsNumpy("pcmparrRawSignalFfted.npy", false, leshape2.size(), leshape2.data(), data4);
		int ii = 0;*/

		cufftComplex* pcmparrRawSignalRolled1 = NULL;
		cudaMalloc(&pcmparrRawSignalRolled1, m_nfft * m_nchan * m_npol / 2 * m_nbin * sizeof(cufftComplex));
		int mbin = get_mbin();
		dim3 threads(256, 1);
		dim3 blocks((mbin + threads.x - 1) / threads.x, m_nfft * m_nchan * m_npol / 2 * m_len_sft);
		// result stored in m_pdcmpbuff_ewmulted
		roll_rows_kernel << < blocks, threads >> > (pcmparrRawSignalRolled1, m_pdcmpbuff_ewmulted, m_nfft * m_nchan * m_npol / 2 * m_len_sft, mbin, mbin / 2);
		cudaDeviceSynchronize();
		// result stored in pcmparrRawSignalRolled1. m_pdcmpbuff_ewmulted is free
		cudaStatus0 = cudaGetLastError();
		if (cudaStatus0 != cudaSuccess) {
			fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus0));
			return false;
		}
		/*int lenarr4 = m_nfft * m_nchan * m_nbin * (m_npol / 2) / 2;
		std::vector<complex<float>> data4(lenarr4, 0);
		cudaMemcpy(data4.data(), pcmparrRawSignalRolled1, lenarr4 * sizeof(std::complex<float>), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		std::array<long unsigned, 1> leshape2{ lenarr4 };
		npy::SaveArrayAsNumpy("pcmparrRawSignalFfted.npy", false, leshape2.size(), leshape2.data(), data4);
		int ii = 0;*/

		checkCudaErrors(cufftExecC2C(m_fftPlanInverse, pcmparrRawSignalRolled1, pcmparrRawSignalRolled1, CUFFT_INVERSE));
		// result stored in pcmparrRawSignalRolled1. m_pdcmpbuff_ewmulted is free
		cudaStatus0 = cudaGetLastError();
		if (cudaStatus0 != cudaSuccess) {
			fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus0));
			return false;
		}		
		

		
		//int lenarr4 = m_nfft * m_nchan * m_nbin * (m_npol / 2) / 2;// *sizeof(cufftComplex));
		//std::vector<complex<float>> data4(lenarr4, 0);
		//cudaMemcpy(data4.data(), &pcmparrRawSignalRolled1[lenarr4], lenarr4 * sizeof(std::complex<float>), cudaMemcpyDeviceToHost);
		//cudaDeviceSynchronize();
		//std::array<long unsigned, 1> leshape2{ lenarr4 };
		//npy::SaveArrayAsNumpy("pcmparrRawSignalFfted.npy", false, leshape2.size(), leshape2.data(), data4);
		//int ii = 0;
		// OK

		int noverlap_per_channel = get_noverlap_per_channel();
		int mbin_adjusted = get_mbin_adjusted();
		cufftComplex* fbuf = m_pdcmpbuff_ewmulted;

		// result stored in pcmparrRawSignalRolled1. m_pdcmpbuff_ewmulted is free
		dim3 threads_per_block1(1024, 1, 1);
		dim3 blocks_per_grid1((mbin_adjusted + threads_per_block1.x - 1) / threads_per_block1.x, m_nchan * m_len_sft, m_nfft * m_npol / 2);
		transpose_unpadd_kernel << < blocks_per_grid1, threads_per_block1 >> >
			(fbuf, pcmparrRawSignalRolled1, m_nfft, noverlap_per_channel
				, mbin_adjusted, m_nchan, m_len_sft, mbin);
		
		
		cudaStatus0 = cudaGetLastError();
		if (cudaStatus0 != cudaSuccess) {
			fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus0));
			return false;
		}
		 //result stored in m_pdcmpbuff_ewmulted (=fbuf).   pcmparrRawSignalRolled1 is free
		
	

		// result stored in m_pdcmpbuff_ewmulted (=fbuf).   pcmparrRawSignalRolled1 is free
		float* parr_wfall = (float*)pcmparrRawSignalRolled1;

		// 
		dim3 threadsPerChunk(TILE_DIM, TILE_DIM, 1);
		dim3 chunkPerGrid((m_nchan * m_len_sft + TILE_DIM - 1) / TILE_DIM, (msamp + TILE_DIM - 1) / TILE_DIM, 1);

		calcPowerMtrx_kernel << <  chunkPerGrid, threadsPerChunk >> > (parr_wfall, msamp, m_nchan * m_len_sft, m_npol, (cufftComplex*)fbuf);

		cudaDeviceSynchronize();
		// result stored in pcmparrRawSignalRolled1.   m_pdcmpbuff_ewmulted  is free
		cudaStatus0 = cudaGetLastError();
		if (cudaStatus0 != cudaSuccess) {
			fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus0));
			return false;

		}
		/*int lenarr4 = msamp * m_nchan * m_len_sft;
		std::vector<float> data4(lenarr4, 0);
		cudaMemcpy(data4.data(), parr_wfall, lenarr4 * sizeof(float), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		std::array<long unsigned, 1> leshape2{ lenarr4 };
		npy::SaveArrayAsNumpy("parr_wfall_py.npy", false, leshape2.size(), leshape2.data(), data4);
		int ii = 0;*/

		// result stored in pcmparrRawSignalRolled1 (=parr_wfall.   m_pdcmpbuff_ewmulted  is free
		float* parr_wfall_disp = (float*)m_pdcmpbuff_ewmulted;
		double val_tsamp_wfall = m_len_sft * m_tsamp;
		double val_dm = m_coh_dm_Vector[idm];
		double f0 = ((double)m_Fmax - (double)m_Fmin) / (m_nchan * m_len_sft);
		dim3 threadsPerblock1(1024, 1, 1);
		dim3 blocksPerGrid1((msamp + threadsPerblock1.x - 1) / threadsPerblock1.x, m_nchan * m_len_sft, 1);
		dedisperse << <  blocksPerGrid1, threadsPerblock1 >> > (parr_wfall_disp, parr_wfall, val_dm
			, (double)m_Fmin, (double)m_Fmax, val_tsamp_wfall, f0, msamp);
		cudaDeviceSynchronize();

		cudaStatus0 = cudaGetLastError();
		if (cudaStatus0 != cudaSuccess) {
			fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus0));
			return false;
		}
		/*int lenarr4 = msamp * m_nchan * m_len_sft;
		std::vector<float> data4(lenarr4, 0);
		cudaMemcpy(data4.data(), parr_wfall_disp, lenarr4 * sizeof(float), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		std::array<long unsigned, 1> leshape2{ lenarr4 };
		npy::SaveArrayAsNumpy("pcmparrRawSignalFfted.npy", false, leshape2.size(), leshape2.data(), data4);
		int ii = 0;*/

		m_Fdmt.process_image(parr_wfall_disp, &m_pdoutImg[idm * msamp * m_len_sft], false);

		cudaStatus0 = cudaGetLastError();
		if (cudaStatus0 != cudaSuccess) {
			fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus0));
			return false;
		}
		cudaFree(pcmparrRawSignalRolled1);
		//cudaFree(fbuf);

		cudaStatus0 = cudaGetLastError();
		if (cudaStatus0 != cudaSuccess) {
			fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus0));
			return false;
		}

	}

	std::vector<std::vector<float>>* pvecImg_temp = nullptr;
	if (nullptr != pvecImg)
	{
		pvecImg->resize(m_coh_dm_Vector.size());
		for (auto& row : *pvecImg)
		{
			row.resize(msamp * m_len_sft);
		}
		pvecImg_temp = pvecImg;
	}
	else
	{
		pvecImg_temp = new std::vector<std::vector<float>>;
		pvecImg_temp->resize(m_coh_dm_Vector.size());
		for (auto& row : *pvecImg_temp)
		{
			row.resize(msamp * m_len_sft);
		}
	}
	for (int i = 0; i < m_coh_dm_Vector.size(); ++i)
	{
		cudaMemcpy(pvecImg_temp->at(i).data(), &m_pdoutImg[i * msamp * m_len_sft], msamp * m_len_sft * sizeof(float), cudaMemcpyDeviceToHost);
	}
	//checkCudaErrors(cudaFree(pcmparrRawSignalRolled));
	//checkCudaErrors(cudaFree(poutImg));
	return true;
}
		


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








