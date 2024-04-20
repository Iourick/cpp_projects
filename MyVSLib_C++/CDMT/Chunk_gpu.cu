#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Chunk_gpu.cuh"
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
	size_t free_bytes, total_bytes;
	cudaError_t cuda_status = cudaMemGetInfo(&free_bytes, &total_bytes);

	extern const unsigned long long TOtal_GPU_Bytes = (long long)free_bytes;

	// timing variables:
	  // fdmt time
	long long iFdmt_time = 0;
	// read && transform data time
	long long  iReadTransform_time = 0;
	// fft time
	long long  iFFT_time = 0;
	// detection time
	long long  iMeanDisp_time = 0;
	// detection time
	long long  iNormalize_time = 0;
	// total time
	long long  iTotal_time = 0;

	CChunk_gpu::~CChunk_gpu()
	{
		if (m_pd_arrcoh_dm)
		{
			cudaFree(m_pd_arrcoh_dm);
		}
		if (m_pd_arr_dc)
		{
			cudaFree(m_pd_arr_dc);
		}

		cufftDestroy(m_fftPlanForward);
		cufftDestroy(m_fftPlanInverse);
	}
	//-----------------------------------------------------------
	CChunk_gpu::CChunk_gpu() :CChunkB()
	{
		m_pd_arrcoh_dm = nullptr;
		m_pd_arr_dc = nullptr;
		
	}
	//-----------------------------------------------------------

	CChunk_gpu::CChunk_gpu(const  CChunk_gpu& R) :CChunkB(R)
	{
		cudaMalloc(&m_pd_arrcoh_dm, R.m_coh_dm_Vector.size() * sizeof(double));
		cudaMemcpy(m_pd_arrcoh_dm, R.m_pd_arrcoh_dm, m_coh_dm_Vector.size() * sizeof(double), cudaMemcpyDeviceToDevice);

		cudaMalloc(&m_pd_arr_dc, R.m_coh_dm_Vector.size() * R.m_nchan * R.m_nbin * sizeof(cufftComplex));
		cudaMemcpy(m_pd_arr_dc, R.m_pd_arr_dc,   R.m_coh_dm_Vector.size() * R.m_nchan * R.m_nbin * sizeof(cufftComplex), cudaMemcpyDeviceToDevice);
		m_Fdmt = R.m_Fdmt;

		cufftDestroy(m_fftPlanForward);
		cufftDestroy(m_fftPlanInverse);
		create_fft_plans();

	}
	//-------------------------------------------------------------------

	CChunk_gpu& CChunk_gpu::operator=(const CChunk_gpu& R)
	{
		if (this == &R)
		{
			return *this;
		}
		CChunkB:: operator= (R);

		cudaMalloc(&m_pd_arrcoh_dm, R.m_coh_dm_Vector.size() * sizeof(double));
		cudaMemcpy(m_pd_arrcoh_dm, R.m_pd_arrcoh_dm, m_coh_dm_Vector.size() * sizeof(double), cudaMemcpyDeviceToDevice);

		cudaMalloc(&m_pd_arr_dc, R.m_coh_dm_Vector.size() * R.m_nchan * R.m_nbin * sizeof(cufftComplex));
		cudaMemcpy(m_pd_arr_dc, R.m_pd_arr_dc, R.m_coh_dm_Vector.size() * R.m_nchan * R.m_nbin * sizeof(cufftComplex), cudaMemcpyDeviceToDevice);

		m_Fdmt = R.m_Fdmt;
		cufftDestroy(m_fftPlanForward);
		cufftDestroy(m_fftPlanInverse);
		create_fft_plans();

		return *this;
	}
	//------------------------------------------------------------------
	CChunk_gpu::CChunk_gpu(
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
		, const float tsamp) : CChunkB(Fmin
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
		// 1.
		const int ndm = m_coh_dm_Vector.size();
		// 1!

		cudaMalloc(&m_pd_arrcoh_dm, ndm * sizeof(double));
		cudaMemcpy(m_pd_arrcoh_dm, m_coh_dm_Vector.data(), ndm * sizeof(double), cudaMemcpyHostToDevice);

		cudaMalloc(&m_pd_arr_dc, ndm * m_nchan * m_nbin * sizeof(cufftComplex));

		compute_chirp_channel( );
	}


void CChunk_gpu::compute_chirp_channel()
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
	kernel_create_arr_dc << < gridSize2, block_Size2 >> > (m_pd_arr_dc, m_pd_arrcoh_dm, d_parr_freqs_chan, d_parr_bin_freqs, d_parr_taper, ndm, m_nchan, m_len_sft, mbin);

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	std::cout << "Time taken by function fncFdmtU_cu: " << duration.count() << " microseconds" << std::endl;
	cudaFree(d_parr_freqs_chan);
	cudaFree(d_parr_taper);
	cudaFree(d_parr_bin_freqs);

	//int lenarr4 = ndm* m_nchan* m_len_sft* mbin;// *sizeof(cufftComplex));
	//std::vector<complex<float>> data4(lenarr4, 0);
	//cudaMemcpy(data4.data(), m_pd_arr_dc, lenarr4 * sizeof(std::complex<float>), cudaMemcpyDeviceToHost);
	//cudaDeviceSynchronize();
	//std::array<long unsigned, 1> leshape2{ lenarr4 };
	//npy::SaveArrayAsNumpy("arr_dc.npy", false, leshape2.size(), leshape2.data(), data4);


	// Generate FFT plans (batch in-place)
	create_fft_plans();


	

}
//--------------------------------------------------------------------------------------------
void CChunk_gpu::create_fft_plans()
{
	if (cufftPlanMany(&m_fftPlanForward,1, &m_nbin,
		NULL, 1, m_nbin, // *inembed, istride, idist
		NULL, 1, m_nbin, // *onembed, ostride, odist
		CUFFT_C2C, m_nfft * m_nchan * m_npol / 2) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: Plan creation failed");
		return;
	}
	int mbin = get_mbin();
	
	checkCudaErrors(cufftPlanMany(&m_fftPlanInverse, 1, &mbin, NULL,            1,      mbin, NULL,           1,         mbin, CUFFT_C2C, m_len_sft * m_nfft * m_nchan * m_npol / 2));
}

//-------------------------------------------------------------------------------------------
__global__
void kernel_create_arr_dc  (cufftComplex* parr_dc, double * parrcoh_dm,double* parr_freqs_chan, double* parr_bin_freqs, double *parr_taper
	, int ndm, int nchan, int len_sft, int mbin)
{
	//__shared__  double temp0[1];
	//__shared__  int i0[1];
	//printf("ququ");
	int ibin = blockIdx.x * blockDim.x + threadIdx.x;
	if (ibin >= mbin)
	{
		return;
	}
	//printf("ququ1");
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
	//printf("val_x = %f ;\n", (float)val_x);
	if (val_x > 0.0)
	{
		//printf("val_x = % ;\n", (float)val_x);
	}
	parr_dc[i0 + ibin].x = float(val_x * parr_taper[ibin]);
	parr_dc[i0 + ibin].y = float(val_y * parr_taper[ibin]);
	
}

//------------------------------------------------------------------------------------------
__global__
void kernel_create_arr_bin_freqs_and_taper(double* d_parr_bin_freqs, double* d_parr_taper,  double  bw_chan,  int mbin)
{	
	int ind = blockIdx.x * blockDim.x + threadIdx.x;
	if (ind >= mbin)
	{
		return;
	}
	double temp = -0.5 * bw_chan + (ind + 0.5) * bw_chan / mbin;
	d_parr_bin_freqs[ind] = temp;
	d_parr_taper[ind] = 1.0 / sqrt(1.0 + pow(temp / (0.47 * bw_chan), 80));
	
}
//----------------------------------------------------------------------------------------
__global__
void kernel_create_arr_freqs_chan(double* d_parr_freqs_chan, int len_sft, double bw_chan, double  Fmin, double bw_sub)
{
	int nchan = gridDim.y;
	int ichan = blockIdx.y;	
	int col_ind = blockIdx.x * blockDim.x + threadIdx.x;
	 double freqs_sub = Fmin + bw_sub * (0.5 + ichan);
	 double vi = (double)(col_ind % len_sft);
	 double temp = bw_chan * (vi - len_sft / 2.0 + 0.5);
	d_parr_freqs_chan[ichan * len_sft + col_ind] = freqs_sub + temp;
}
	////---------------------------------------------------
	bool CChunk_gpu::process(void* pcmparrRawSignalCur
		, std::vector<COutChunkHeader>* pvctSuccessHeaders, std::vector<std::vector<float>>* pvecImg)
	{
		// 1. 
		const int mbin = get_mbin();
		const int noverlap_per_channel = get_noverlap_per_channel();
		const int mbin_adjusted = get_mbin_adjusted();
		const int msamp = get_msamp();
		const int mchan = m_nchan * m_len_sft;
		// 1!
		
		// 2. Forward FFT execution
		checkCudaErrors(cufftExecC2C(m_fftPlanForward, (cufftComplex*)pcmparrRawSignalCur, (cufftComplex*)pcmparrRawSignalCur, CUFFT_FORWARD));

		
		//2!	

		//int lenarr4 = m_nfft * m_nchan * m_nbin * (m_npol / 2) / 2;// *sizeof(cufftComplex));
		//std::vector<complex<float>> data4(lenarr4, 0);
		//cudaMemcpy(data4.data(), (cufftComplex*)pcmparrRawSignalCur, lenarr4 * sizeof(std::complex<float>), cudaMemcpyDeviceToHost);
		//cudaDeviceSynchronize();
		//std::array<long unsigned, 1> leshape2{ lenarr4 };
		//npy::SaveArrayAsNumpy("pcmparrRawSignalFfted.npy", false, leshape2.size(), leshape2.data(), data4);
		//int ii = 0;

		//3. roll and normalize ffted signals
		dim3 treads_per_block(256, 1);
		dim3 blocks_per_grid((m_nbin + treads_per_block.x - 1) / treads_per_block.x, m_nfft * m_nchan * m_npol / 2);
		
		cufftComplex* pcmparrRawSignalRolled = NULL;
		cudaMalloc(&pcmparrRawSignalRolled, m_nfft * m_nchan * m_npol / 2 * m_nbin * sizeof(cufftComplex));
		dim3 threads(1024, 1);
		dim3 blocks((m_nbin + threads.x - 1) / threads.x, m_nfft * m_nchan * m_npol / 2);
		roll_rows_and_normalize_kernel<<< blocks, threads>>>(pcmparrRawSignalRolled, (cufftComplex*)pcmparrRawSignalCur, m_nfft * m_nchan * m_npol / 2, m_nbin, m_nbin / 2);

		//int lenarr4 = m_nfft * m_nchan * m_nbin * (m_npol / 2) / 2;// *sizeof(cufftComplex));
		//std::vector<complex<float>> data4(lenarr4, 0);
		//cufftComplex* pc = &pcmparrRawSignalRolled[lenarr4];
		//cudaMemcpy(data4.data(), pcmparrRawSignalRolled, lenarr4 * sizeof(std::complex<float>), cudaMemcpyDeviceToHost);
		//cudaDeviceSynchronize();
		//std::array<long unsigned, 1> leshape2{ lenarr4 };
		//npy::SaveArrayAsNumpy("pcmparrRawSignalFfted.npy", false, leshape2.size(), leshape2.data(), data4);
		//int ii = 0;
		
		//3!
		for (int idm = 0; idm < m_coh_dm_Vector.size(); ++idm)
		{
			dim3 threadsPerBlock(1024, 1, 1);
			dim3 blocksPerGrid((m_nchan * m_nbin + threadsPerBlock.x - 1) / threadsPerBlock.x, m_nfft, m_npol/2);
			element_wise_cufftComplex_mult_kernel<<< blocksPerGrid, threadsPerBlock>>>
				((cufftComplex*)pcmparrRawSignalCur, pcmparrRawSignalRolled, &m_pd_arr_dc[m_nchan * m_nbin * idm], m_npol/2,m_nfft, m_nchan * m_nbin);


			 int threads1 = 1024;
			int blocks1 = (m_npol / 2 * m_nfft * m_nchan * m_nbin + threads1 - 1) / threads1;
			
			divide_cufftComplex_array_kernel <<<blocks1, threads1>>>((cufftComplex*)pcmparrRawSignalCur, m_npol / 2* m_nfft * m_nchan * m_nbin,  ((float)m_nbin));
		//	int lenarr4 = m_nfft * m_nchan * m_nbin * (m_npol / 2) / 2;// *sizeof(cufftComplex));
		//std::vector<complex<float>> data4(lenarr4, 0);
		//cudaMemcpy(data4.data(), (cufftComplex*)pcmparrRawSignalCur, lenarr4 * sizeof(std::complex<float>), cudaMemcpyDeviceToHost);
		//cudaDeviceSynchronize();
		//std::array<long unsigned, 1> leshape2{ lenarr4 };
		//npy::SaveArrayAsNumpy("pcmparrRawSignalFfted.npy", false, leshape2.size(), leshape2.data(), data4);
		//int ii = 0;
			cufftComplex* pcmparrRawSignalRolled1 = NULL;
			cudaMalloc(&pcmparrRawSignalRolled1, m_nfft * m_nchan * m_npol / 2 * m_nbin * sizeof(cufftComplex));
			int mbin = get_mbin();
			dim3 threads(256, 1);
			dim3 blocks((mbin + threads.x - 1) / threads.x, m_nfft * m_nchan * m_npol / 2 * m_len_sft);
			roll_rows_and_normalize_kernel << < blocks, threads >> > (pcmparrRawSignalRolled1, (cufftComplex*)pcmparrRawSignalCur, m_nfft * m_nchan * m_npol / 2 *  m_len_sft, mbin, mbin);

			//int lenarr4 = m_nfft * m_nchan * m_nbin * (m_npol / 2) / 2;// *sizeof(cufftComplex));
			//std::vector<complex<float>> data4(lenarr4, 0);
			//cudaMemcpy(data4.data(), pcmparrRawSignalRolled1, lenarr4 * sizeof(std::complex<float>), cudaMemcpyDeviceToHost);
			//cudaDeviceSynchronize();
			//std::array<long unsigned, 1> leshape2{ lenarr4 };
			//npy::SaveArrayAsNumpy("pcmparrRawSignalFfted.npy", false, leshape2.size(), leshape2.data(), data4);
			//int ii = 0;

			checkCudaErrors(cufftExecC2C(m_fftPlanInverse, pcmparrRawSignalRolled1, pcmparrRawSignalRolled1, CUFFT_INVERSE));
			//checkCudaErrors(cufftExecC2C(m_fftPlanInverse, (cufftComplex*)pcmparrRawSignalCur, (cufftComplex*)pcmparrRawSignalCur, CUFFT_INVERSE));
			
			
			blocks1 = (m_nfft * m_nchan * m_npol / 2 * m_nbin + threads1 - 1) / threads1;
			divide_cufftComplex_array_kernel << <blocks1, threads1 >> > (pcmparrRawSignalRolled1, m_nfft * m_nchan * m_npol / 2 * m_nbin, ((float)mbin));

			//int lenarr4 = m_nfft * m_nchan * m_nbin * (m_npol / 2) / 2;// *sizeof(cufftComplex));
			//std::vector<complex<float>> data4(lenarr4, 0);
			//cudaMemcpy(data4.data(), pcmparrRawSignalRolled1, lenarr4 * sizeof(std::complex<float>), cudaMemcpyDeviceToHost);
			//cudaDeviceSynchronize();
			//std::array<long unsigned, 1> leshape2{ lenarr4 };
			//npy::SaveArrayAsNumpy("pcmparrRawSignalFfted.npy", false, leshape2.size(), leshape2.data(), data4);
			//int ii = 0;


			int noverlap_per_channel = get_noverlap_per_channel();
			int mbin_adjusted = get_mbin_adjusted();
			void* fbuf = NULL;
			cudaMalloc(&fbuf, mbin_adjusted * m_nfft* m_nchan * m_len_sft * m_npol / 2 * sizeof(cufftComplex));
			//void* fbuf = (cufftComplex*)pcmparrRawSignalCur;
			
			dim3 threads_per_block1(1024, 1, 1);
			dim3 blocks_per_grid1(( mbin_adjusted + threads_per_block1.x - 1) / threads_per_block1.x, m_nchan * m_len_sft, m_nfft * m_npol/2);
			transpose_unpadd<<< blocks_per_grid1, threads_per_block1>>>
				((cufftComplex*)fbuf, pcmparrRawSignalRolled1, m_nfft, noverlap_per_channel
				, mbin_adjusted, m_nchan, m_len_sft, mbin);

			int lenarr4 = m_nfft * mbin_adjusted * m_nchan * m_len_sft * m_npol / 2 / 2;// *sizeof(cufftComplex));
			std::vector<complex<float>> data4(lenarr4, 0);
			cudaMemcpy(data4.data(), fbuf, lenarr4 * sizeof(std::complex<float>), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
			std::array<long unsigned, 1> leshape2{ lenarr4 };
			npy::SaveArrayAsNumpy("pcmparrRawSignalFfted.npy", false, leshape2.size(), leshape2.data(), data4);
			int ii = 0;

			cudaFree(pcmparrRawSignalRolled1);
			cudaFree(fbuf);
		}
		cudaFree(pcmparrRawSignalRolled);
		return true;
	}
//-----------------------------------------------------------------------------------
//		
__global__	void  transpose_unpadd(cufftComplex* fbuf, cufftComplex* arin,int nfft,  int noverlap_per_channel
	, int mbin_adjusted, const int nchan, const int nlen_sft, int mbin)	
{
	int  ibin = blockIdx.x * blockDim.x + threadIdx.x;
	if (!(ibin < mbin_adjusted))
	{
		return;
	}
	int ipol = blockIdx.z / nfft;
	int ifft = blockIdx.z % nfft;
	int ichan = blockIdx.y / nlen_sft;
	int ilen_sft = blockIdx.y % nlen_sft;
	int ibin_adjusted = ibin + noverlap_per_channel;
	int isamp = ibin + mbin_adjusted * ifft;
	 
	
	// Select bins from valid region and reverse the frequency axis		
	// float temp = arin[ifft * nchan * nlen_sft * mbin + (nchan - ichan - 1) * nlen_sft * mbin + ilen_sft * mbin + ibin_adjusted].y;
	 //printf("%f \n", temp);
	
	fbuf[ipol * mbin_adjusted * nchan * nlen_sft  +  isamp * nchan * nlen_sft + ichan * nlen_sft + nlen_sft - ilen_sft - 1].x =
		arin[ipol * nfft * nchan * nlen_sft * mbin + ifft * nchan * nlen_sft * mbin + (nchan - ichan - 1) * nlen_sft * mbin + ilen_sft * mbin + ibin_adjusted].x;
	fbuf[ipol * mbin_adjusted * nchan * nlen_sft +  isamp * nchan * nlen_sft + ichan * nlen_sft + nlen_sft - ilen_sft - 1].y =
		arin[ipol *  nfft * nchan * nlen_sft * mbin +  ifft * nchan * nlen_sft * mbin + (nchan - ichan - 1) * nlen_sft * mbin + ilen_sft * mbin + ibin_adjusted].y;
	//for (int ifft = 0; ifft < nfft; ++ifft)
	//{
	//	for (int ichan = 0; ichan < nchan; ++ichan)
	//	{
	//		for (int ilen_sft = 0; ilen_sft < nlen_sft; ++ilen_sft)
	//		{
	//			for (int ibin = 0; ibin < mbin_adjusted; ++ibin)
	//			{
	//				int ibin_adjusted = ibin + noverlap_per_channel;
	//				int isamp = ibin + mbin_adjusted * ifft;


	//				// Select bins from valid region and reverse the frequency axis					
	//				fbuf[isamp * nchan * nlen_sft + ichan * nlen_sft + nlen_sft - ilen_sft - 1][0] =
	//					arin[ifft * nchan * nlen_sft * mbin + (nchan - ichan - 1) * nlen_sft * mbin + ilen_sft * mbin + ibin_adjusted][0];
	//				fbuf[isamp * nchan * nlen_sft + ichan * nlen_sft + nlen_sft - ilen_sft - 1][1] =
	//					arin[ifft * nchan * nlen_sft * mbin + (nchan - ichan - 1) * nlen_sft * mbin + ilen_sft * mbin + ibin_adjusted][1];


	//			}
	//		}
	//	}
	//}
}
	
//
//	void  CChunk_cpu::transpose_unpadd(fftwf_complex* arin, fftwf_complex* fbuf)
//	{
//		int noverlap_per_channel = get_noverlap_per_channel();
//		int mbin_adjusted = get_mbin_adjusted();
//		const int nsub = m_nchan;
//		const int nchan = m_len_sft;
//		const int mbin = get_mbin();
//#pragma omp parallel
//		{
//			for (int ifft = 0; ifft < m_nfft; ++ifft)
//			{
//				for (int ichan = 0; ichan < nchan; ++ichan)
//				{
//					for (int ibin = 0; ibin < mbin_adjusted; ++ibin)
//					{
//						int ibin_adjusted = ibin + noverlap_per_channel;
//						int isamp = ibin + mbin_adjusted * ifft;
//						int num = 0;
//						for (int isub = 0; isub < nsub; ++isub)
//						{
//							// Select bins from valid region and reverse the frequency axis					
//							fbuf[isamp * nsub * nchan + isub * nchan + nchan - ichan - 1][0] =
//								arin[ifft * nsub * nchan * mbin + (nsub - isub - 1) * nchan * mbin + ichan * mbin + ibin_adjusted][0];
//							fbuf[isamp * nsub * nchan + isub * nchan + nchan - ichan - 1][1] =
//								arin[ifft * nsub * nchan * mbin + (nsub - isub - 1) * nchan * mbin + ichan * mbin + ibin_adjusted][1];
//							++num;
//						}
//					}
//				}
//			}
//		}
//	}
	//-----------------------------------------------------------------------
	__global__ void  divide_cufftComplex_array_kernel(cufftComplex* d_arr, int len, float val)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= len)
		{
			return;
		}
		d_arr[idx].x /= val;
		d_arr[idx].y /= val;
	}
//-------------------------------------------------------------------------------------------------
	//---------------------------------------------------------------
	__global__
		void scaling_kernel(cufftComplex* data, long long element_count, float scale)
	{
		const int tid = threadIdx.x;
		const int stride = blockDim.x;
		for (long long i = tid; i < element_count; i += stride)
		{
			data[i].x *= scale;
			data[i].y *= scale;
		}
	}
__global__ void  element_wise_cufftComplex_mult_kernel(cufftComplex * d_arrOut, cufftComplex * d_arrInp0, cufftComplex * d_arrInp1
	, int npol, int nfft, int dim2)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= dim2)
	{
		return;
	}
	//printf("ff\n");
	int ipol = blockIdx.z;
	int ifft = blockIdx.y;
	int ibegin = ipol * nfft * dim2 + ifft * dim2;
	d_arrOut[ibegin + idx].x = d_arrInp0[ibegin + idx].x* d_arrInp1[idx].x - d_arrInp0[ibegin + idx].y * d_arrInp1[idx].y;
	d_arrOut[ibegin + idx].y = d_arrInp0[ibegin + idx].x* d_arrInp1[idx].y + d_arrInp0[ibegin + idx].y * d_arrInp1[idx].x;
	//printf("%f  %f\n", d_arrOut[ibegin + idx].x, d_arrOut[ibegin + idx].y);
	
}
	
	//--------------------------------------------------------------------------------

	__global__ void roll_rows_and_normalize_kernel(cufftComplex* arr_rez, cufftComplex* arr, int rows, int cols, int shift)
	{

		int idx0 = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx0 >= cols)
		{
			return;
		}
		int ind_new = blockIdx.y * cols + (idx0 + shift) % cols;
		int ind = blockIdx.y * cols + idx0;
		arr_rez[ind_new].x = arr[ind].x /*/cols*/;
		arr_rez[ind_new].y = arr[ind].y/*/cols*/ ;

	}

//-----------------------------------------------------------------

long long CChunk_gpu::calcLenChunk_(CTelescopeHeader header, const int nsft
	, const float pulse_length, const float d_max)
{
	const int nchan_actual = nsft * header.m_nchan;

	long long len = 0;
	for (len = 1 << 9; len < 1 << 30; len <<= 1)
	{
		// iMaxDt !!!!! ???????????????? 
		CFdmtGpu fdmt(
			header.m_centfreq - header.m_chanBW * header.m_nchan / 2.
			, header.m_centfreq + header.m_chanBW * header.m_nchan / 2.
			, nchan_actual
			, len
			, nchan_actual
		);

		
		long long size0 = fdmt.calcSizeAuxBuff_fdmt_();
		long long size_fdmt_inp = fdmt.calc_size_input();
		long long size_fdmt_out = fdmt.calc_size_output();
		long long size_fdmt_norm = size_fdmt_out;
		long long irest = header.m_nchan * header.m_npol * header.m_nbits / 8 // input buff
			+ header.m_nchan * header.m_npol / 2 * sizeof(cufftComplex)
			+ 3 * header.m_nchan * header.m_npol * sizeof(cufftComplex) / 2
			+ 2 * header.m_nchan * sizeof(float);
		irest *= len;

		long long rez = size0 + size_fdmt_inp + size_fdmt_out + size_fdmt_norm + irest;
		if (rez > 0.98 * TOtal_GPU_Bytes)
		{
			return len / 2;
		}

	}
	return -1;
}

//---------------------------------------------------
//bool CChunk_gpu::fnCChunk_gpuProcessing_gpu(cufftComplex* pcmparrRawSignalCur
//	, void* pAuxBuff_fdmt
//	, cufftComplex* pcarrTemp
//	, cufftComplex* pcarrCD_Out
//	, cufftComplex* pcarrBuff
//	, float* pInpOutBuffFdmt, fdmt_type_* d_arrfdmt_norm
//	, const int IDeltaT, cufftHandle plan0, cufftHandle plan1
//	, structOutDetection* pstructOut
//	, float* pcoherentDedisp)
//{
//	// 1. installation of pointers	for pAuxBuff_the_rest
//	fdmt_type_* d_parr_fdmt_inp = (fdmt_type_*)pInpOutBuffFdmt; //4	
//	fdmt_type_* d_parr_fdmt_out = (fdmt_type_*)pInpOutBuffFdmt + m_nbin * m_nchan;
//	//// !1
//
//	 /*std::vector<std::complex<float>> data2(m_nbin, 0);
//	cudaMemcpy(data2.data(), pcmparrRawSignalCur, m_nbin * sizeof(std::complex<float>),
//		cudaMemcpyDeviceToHost);
//	cudaDeviceSynchronize();*/
//	//std::array<long unsigned, 1> leshape127{ LEnChunk };
//	//npy::SaveArrayAsNumpy("ffted.npy", false, leshape127.size(), leshape127.data(), data);
//	auto start = std::chrono::high_resolution_clock::now();
//
//
//	// 2. create FFT	
//	//cufftComplex* pcmparr_ffted = NULL;
//	//checkCudaErrors(cudaMallocManaged((void**)&pcmparr_ffted, m_nbin * m_nchan * m_npol/2 * sizeof(cufftComplex)));
//	//checkCudaErrors(cufftExecC2C(plan0, pcmparrRawSignalCur, pcmparr_ffted, CUFFT_FORWARD));
//	checkCudaErrors(cufftExecC2C(plan0, pcmparrRawSignalCur, pcmparrRawSignalCur, CUFFT_FORWARD));
//
//	// !2
//
//	auto end = std::chrono::high_resolution_clock::now();
//	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
//	iFFT_time += duration.count();
//
//	/*std::vector<std::complex<float>> data(m_nbin * m_nchan * m_npol/2, 0);
//	cudaMemcpy(data.data(), pcmparrRawSignalCur, m_nbin * m_nchan * m_npol/2 * sizeof(std::complex<float>),
//		cudaMemcpyDeviceToHost);
//	cudaDeviceSynchronize();*/
//
//
//	// 3.
//	/*float chanBW = (m_Fmax - m_Fmin) / m_nchan;
//	float f1 = chanBW + m_Fmin;
//	float valConversionConst = DISPERSION_CONSTANT * (1. / (m_Fmin * m_Fmin) - 1. / (f1 * f1)) * chanBW;*/
//	float valConversionConst = DISPERSION_CONSTANT * (1. / (m_Fmin * m_Fmin) - 1. / (m_Fmax * m_Fmax)) * (m_Fmax - m_Fmin);
//	float valN_d = m_d_max * valConversionConst;
//	const int N_p = m_len_sft * m_nchan;
//	int n_coherent = int(ceil(valN_d / (N_p * N_p)));
//	cout << " n_coherent = " << n_coherent << endl;
//	// !3
//
//
//	structOutDetection* pstructOutCur = NULL;
//	checkCudaErrors(cudaMallocManaged((void**)&pstructOutCur, sizeof(structOutDetection)));
//	cudaDeviceSynchronize();
//	pstructOutCur->snr = 1. - FLT_MAX;
//	pstructOutCur->icol = -1;
//	pstructOut->snr = m_sigma_bound;
//	//// 4. main loop
//	const int IMaxDT = N_p;
//
//	float coherent_d = -1.;
//	bool breturn = false;
//	for (int iouter_d = 0; iouter_d < n_coherent; ++iouter_d)
//		//for (int iouter_d = 31; iouter_d < 32; ++iouter_d)
//	{
//
//		 float valcur_coherent_d = ((float)iouter_d) * ((float)m_d_max / ((float)n_coherent));
//		cout << "Chunk=  " << m_Chunk_id << "; chunk= " << iouter_d << "; iter= " << iouter_d << "; coherent_d = " << valcur_coherent_d << endl;
//
//		/*std::vector<std::complex<float>> data(m_nbin * m_nchan * m_npol /2, 0);
//		cudaMemcpy(data.data(), pcmparrRawSignalCur, m_nbin * m_nchan * m_npol / 2 * sizeof(std::complex<float>),
//		cudaMemcpyDeviceToHost);
//		cudaDeviceSynchronize();*/
//
//		// fdmt input matrix computation
//		calcFDMT_Out_gpu(d_parr_fdmt_out, pcmparrRawSignalCur, pcarrCD_Out
//			, pcarrTemp, d_parr_fdmt_inp
//			, IMaxDT, DISPERSION_CONSTANT * valcur_coherent_d
//			, pAuxBuff_fdmt, IDeltaT, plan0, plan1, pcarrBuff);
//		// !
//		///*float* parr_fdmt_out = (float*)malloc(m_nbin * m_nchan* sizeof(float));
//		//cudaMemcpy(parr_fdmt_out, d_parr_fdmt_out, m_nbin * m_nchan * sizeof(float), cudaMemcpyDeviceToHost);
//		//float valmax = -0., valmin = 0.;
//		//unsigned int iargmax = -1, iargmin = -1;
//		//findMaxMinOfArray(parr_fdmt_out, m_nbin * m_nchan, &valmax, &valmin
//		//	, &iargmax, &iargmin);
//		//float* arrfdmt_norm = (float*)malloc(m_nbin * m_nchan * sizeof(float));
//		//cudaMemcpy(arrfdmt_norm, d_arrfdmt_norm, m_nbin * m_nchan * sizeof(float), cudaMemcpyDeviceToHost);
//		//float valmax1 = -0., valmin1 = 0.;
//		//unsigned int iargmax1 = -1, iargmin1 = -1;
//		//findMaxMinOfArray(arrfdmt_norm, m_nbin * m_nchan, &valmax1, &valmin1
//		//	, &iargmax1, &iargmin1);
//		//free(parr_fdmt_out);
//		//free(arrfdmt_norm);*/
//
//		//
//		const int Rows = m_len_sft * m_nchan;
//		const int Cols = m_nbin / m_len_sft;
//		const dim3 Totalsize(1024, 1, 1);
//		const dim3 gridSize((Cols + Totalsize.x - 1) / Totalsize.x, Rows, 1);
//		float* d_pAuxArray = (float*)d_parr_fdmt_inp;
//		int* d_pAuxNumArray = (int*)(d_pAuxArray + gridSize.x * gridSize.y);
//		int* d_pWidthArray = d_pAuxNumArray + gridSize.x * gridSize.y;
//		detect_signal_gpu(d_parr_fdmt_out, d_arrfdmt_norm, Rows
//			, Cols, m_length_sum_wnd, gridSize, Totalsize
//			, d_pAuxArray, d_pAuxNumArray, d_pWidthArray, pstructOutCur);
//		if ((*pstructOutCur).snr >= (*pstructOut).snr)
//		{
//			(*pstructOut).snr = (*pstructOutCur).snr;
//			(*pstructOut).icol = (*pstructOutCur).icol;
//			(*pstructOut).irow = (*pstructOutCur).irow;
//			(*pstructOut).iwidth = (*pstructOutCur).iwidth;
//
//			*pcoherentDedisp = valcur_coherent_d;
//
//			std::cout << "SNR = " << (*pstructOut).snr << endl;
//			std::cout << "ROW ARGMAX = " << (*pstructOut).irow << endl;
//			std::cout << "COLUMN ARGMAX = " << (*pstructOut).icol << endl;
//
//			int frequency = 1500; // Frequency in hertz
//			int duration = 500;   // Duration in milliseconds
//			emitSound(frequency, duration / 4);
//			emitSound(frequency + 500, duration / 2);
//			d_pAuxArray = NULL;
//			d_pAuxNumArray = NULL;
//
//			d_pWidthArray = NULL;
//
//			breturn = true;
//		}
//		//
//		///*std::vector<float> data(LEnChunk, 0);
//		//cudaMemcpy(data.data(), parr_fdmt_out, LEnChunk * sizeof(float),
//		//	cudaMemcpyDeviceToHost);
//		//cudaDeviceSynchronize();*/
//	}
//	cudaFree(pstructOutCur);
//	return breturn;
//}
//--------------------------------------------------------------

//--------------------------------------------------------------------
//INPUT:
//1. pcarrTemp - complex array with total length  = m_nbin * (m_npol/2)* m_nchan
// pcarrTemp can be interpreted as matrix, consisting of  m_nchan *(m_npol/2) rows
// each row consists of m_len_sft subrows corresponding to m_len_sft subfrequencies
// 2.pAuxBuff - auxillary buffer to compute mean and dispersions of each row ofoutput matrix d_parr_fdmt_inp
//OUTPUT:
//d_parr_fdmt_inp - matrix with dimensions (m_nchan*m_len_sft) x (m_nbin/m_len_sft)
// d_parr_fdmt_inp[i][j] = 
//
void CChunk_gpu::calc_fdmt_inp(fdmt_type_* d_parr_fdmt_inp, cufftComplex* pcarrTemp
	, float*pAuxBuff)
{	
	
	/*dim3 threadsPerChunk(TILE_DIM, TILE_DIM, 1);
	dim3 ChunksPerGrid((m_len_sft + TILE_DIM - 1) / TILE_DIM, (m_nbin / m_len_sft + TILE_DIM - 1) / TILE_DIM, m_nchan);
	size_t sz = TILE_DIM * (TILE_DIM + 1) * sizeof(float);
	float* d_parr_fdmt_inp_flt = pAuxBuff;	
	calcPowerMtrx_kernel << < ChunksPerGrid, threadsPerChunk, sz >> > (d_parr_fdmt_inp_flt, m_nbin/ m_len_sft, m_len_sft, m_npol, pcarrTemp);
	cudaDeviceSynchronize();*/


	//std::vector<float> data0(m_nbin, 0);
	//cudaMemcpy(data0.data(), d_parr_fdmt_inp_flt, m_nbin * sizeof(float),
	//	cudaMemcpyDeviceToHost);
	//cudaDeviceSynchronize();
	//float valmax = 0., valmin = 0.;
	//unsigned int iargmax = 0, iargmin = 0;
	//findMaxMinOfArray(data0.data(), data0.size(), &valmax, &valmin
	//	, &iargmax, &iargmin);
	float* d_parr_fdmt_temp = pAuxBuff;
	dim3 threadsPerChunk(1024, 1, 1);
	dim3 ChunksPerGrid((m_nbin * threadsPerChunk.x - 1) / threadsPerChunk.x, m_nchan, 1);
	calcPartSum_kernel<<< ChunksPerGrid, threadsPerChunk>>>(d_parr_fdmt_temp, m_nbin, m_npol/2, pcarrTemp);
	cudaDeviceSynchronize();

	/*std::vector<float> data0(m_nbin, 0);
	cudaMemcpy(data0.data(), d_parr_fdmt_temp, m_nbin * sizeof(float),
		cudaMemcpyDeviceToHost);*/

	float* d_parr_fdmt_inp_flt = d_parr_fdmt_temp + m_nbin * m_nchan;
	dim3 threadsPerBlock1(TILE_DIM, TILE_DIM, 1);
	dim3 blocksPerGrid1((m_len_sft + TILE_DIM - 1) / TILE_DIM, (m_nbin / m_len_sft + TILE_DIM - 1) / TILE_DIM, m_nchan);
	size_t sz = TILE_DIM * (TILE_DIM + 1) * sizeof(float);
	multiTransp_kernel << < blocksPerGrid1, threadsPerBlock1, sz >> > (d_parr_fdmt_inp_flt, m_nbin / m_len_sft, m_len_sft, d_parr_fdmt_temp);
	cudaDeviceSynchronize();

	/*std::vector<float> data6(m_nbin, 0);
	cudaMemcpy(data6.data(), d_parr_fdmt_inp_flt, m_nbin * sizeof(float),
		cudaMemcpyDeviceToHost);*/
	int nFdmtRows = m_nchan * m_len_sft;
	int nFdmtCols = m_nbin / m_len_sft;
	float* d_arrRowMean = (float*)pcarrTemp;
	float* d_arrRowDisp = d_arrRowMean + nFdmtRows;
	
	
	auto start = std::chrono::high_resolution_clock::now();

	// Calculate mean and variance
	float* pval_mean = d_arrRowDisp + nFdmtRows;
	float* pval_stdDev = pval_mean + 1;
	float* pval_dispMean = pval_stdDev + 1;
	float* pval_dispStd = pval_dispMean + 1;
	

	ChunksPerGrid = nFdmtRows;
	int treadsPerChunk = calcThreadsForMean_and_Disp(nFdmtCols);
	size_t sz1 = (2 * sizeof(float) + sizeof(int)) * treadsPerChunk;
	// 1. calculations mean values and dispersions for each row of matrix d_parr_fdmt_inp_flt
	// d_arrRowMean - array contents  mean values of each row of input matrix pcarrTemp
	// d_arrRowDisp - array contents  dispersions of each row of input matrix pcarrTemp
	
	calcRowMeanAndDisp << < ChunksPerGrid, treadsPerChunk, sz1 >> > (d_parr_fdmt_inp_flt, nFdmtRows, nFdmtCols, d_arrRowMean, d_arrRowDisp);
	cudaDeviceSynchronize();

	/*std::vector<float> data4(nRows, 0);
	cudaMemcpy(data4.data(), d_arrRowDisp, nRows * sizeof(float),
		cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	std::vector<float> data5(nRows, 0);
	cudaMemcpy(data5.data(), d_arrRowMean, nRows * sizeof(float),
		cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();*/

	//float* parr_fdmt_inp_flt = (float*)malloc(nRows* nCols *sizeof(float));
	//cudaMemcpy(parr_fdmt_inp_flt, d_parr_fdmt_inp_flt, nRows * nCols * sizeof(float), cudaMemcpyDeviceToHost);
	//float* arrM = (float*)malloc(nRows * sizeof(float));
	//float* arrD = (float*)malloc(nRows * sizeof(float));
	//memset(arrM, 0, nRows * sizeof(float));
	//memset(arrD, 0, nRows * sizeof(float));
	//for (int i = 0; i < nRows; ++i)
	//{
	//	for (int j = 0; j < nCols; ++j)
	//	{
	//		arrM[i] += parr_fdmt_inp_flt[i * nCols + j];
	//		arrD[i] += parr_fdmt_inp_flt[i * nCols + j] * parr_fdmt_inp_flt[i * nCols + j];
	//	}
	//	arrM[i] = arrM[i] / ((float)nCols);
	//	arrD[i] = arrD[i] / ((float)nCols) - arrM[i] * arrM[i];

	//}


	//free(parr_fdmt_inp_flt);
	//free(arrM);
	//free(arrD);
	// 2. calculations mean value and standart deviation for full matrix pcarrTemp
	// it is demanded to normalize matrix pcarrTemp
	ChunksPerGrid = 1;
	treadsPerChunk = calcThreadsForMean_and_Disp(nFdmtRows);
	sz = treadsPerChunk * (2 * sizeof(float) + sizeof(int));
	kernel_OneSM_Mean_and_Std << <ChunksPerGrid, treadsPerChunk, sz >> > (d_arrRowMean, d_arrRowDisp, nFdmtRows
		, pval_mean, pval_stdDev);
	cudaDeviceSynchronize();

	//

	float mean = -1., disp = -1.;
	cudaMemcpy(&mean, pval_mean, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&disp, pval_stdDev, sizeof(float), cudaMemcpyDeviceToHost);

	//// check up
	//float* arrmean = (float*)malloc(nRows * sizeof(float));
	//float* arrdisp = (float*)malloc(nRows * sizeof(float));
	//cudaMemcpy(arrmean, d_arrRowMean, nRows * sizeof(float), cudaMemcpyDeviceToHost);
	//cudaMemcpy(arrdisp, d_arrRowDisp, nRows * sizeof(float), cudaMemcpyDeviceToHost);
	//float sum = 0.;
	//for (int i = 0; i < nRows; ++i)
	//{
	//	sum += arrmean[i];
	//}
	//sum = sum / ((float)nRows);

	//float disp1 = 0;
	//for (int i = 0; i < nRows; ++i)
	//{
	//	disp1 += arrmean[i] * arrmean[i] + arrdisp[i];// (arrmean[i] - sum)* (arrmean[i] - sum);
	//}
	//disp1 = disp1/ ((float)nRows) - sum*sum;

	//free(arrmean);
	//free(arrdisp);


	// 3. calculations mean value and standart deviation for array d_arrRowDisp
	// it is demanded to clean out tresh from matrix pcarrTemp
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	iMeanDisp_time += duration.count();
	
	
	
	int threads = 128;
	calculateMeanAndSTD_for_oneDimArray_kernel << <1, threads, threads * 2 * sizeof(float) >> > (d_arrRowDisp, nFdmtRows, pval_dispMean, pval_dispStd);
	cudaDeviceSynchronize();

	/*float hval_dispMean = -1;
	float hval_dispStd = -1;
	cudaMemcpy(&hval_dispMean, pval_dispMean, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&hval_dispStd, pval_dispStd, sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	pval_dispMean = NULL;
	pval_dispStd = NULL;*/

	// 4.Clean and normalize array
	const dim3 Totalsize(256, 1, 1);
	const dim3 gridSize(1, nFdmtRows, 1);
	
	

	normalize_and_clean << < gridSize, Totalsize >> >
		(d_parr_fdmt_inp, d_parr_fdmt_inp_flt, nFdmtRows, nFdmtCols
		, pval_mean, pval_stdDev, d_arrRowDisp, pval_dispMean, pval_dispStd  );	
	cudaDeviceSynchronize();

	float* parr_fdmt_inp = (float*)malloc(nFdmtRows * nFdmtCols * sizeof(float));
	cudaMemcpy(parr_fdmt_inp, d_parr_fdmt_inp, nFdmtRows* nFdmtCols * sizeof(float), cudaMemcpyDeviceToHost);

	//float valmax = -0., valmin = 0.;
	//unsigned int iargmax = -1, iargmin = -1;
	//findMaxMinOfArray(parr_fdmt_inp, nRows * nCols, &valmax,  &valmin
	//	, &iargmax, &iargmin);

	//auto end1 = std::chrono::high_resolution_clock::now();
	//auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - end);
	//iNormalize_time += duration1.count();
	//free(parr_fdmt_inp);

	d_arrRowMean = NULL;
	d_arrRowDisp = NULL;
	pval_mean = NULL;	
	pval_stdDev = NULL;
}
//--------------------------------------
void CChunk_gpu::set_chunkid(const int nC)
{
	m_Chunk_id = nC;
}
//--------------------------------------
void CChunk_gpu::set_blockid(const int nC)
{
	m_Block_id = nC;
}
//-------------------------------------------------------------------
__device__
float fnc_norm2(cufftComplex* pc)
{
	return ((*pc).x * (*pc).x + (*pc).y * (*pc).y);
}




//----------------------------------------------------

__global__
void calcMultiTransposition_kernel(fdmt_type_* output, const int height, const int width, fdmt_type_* input)
{
	__shared__ fdmt_type_ tile[TILE_DIM][TILE_DIM + 1]; // Shared memory tile
	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;
	int ichan = blockIdx.z;
	// Transpose data from global to shared memory
	if (x < width && y < height)
	{
		tile[threadIdx.y][threadIdx.x] = input[ichan * height * width + y * width + x];
	}
	__syncthreads();

	// Calculate new indices for writing to output
	x = blockIdx.y * TILE_DIM + threadIdx.x;
	y = blockIdx.x * TILE_DIM + threadIdx.y;

	// Transpose data from shared to global memory
	if (x < height && y < width)
	{
		output[ichan * height * width + y * height + x] = tile[threadIdx.x][threadIdx.y];
	}
}
//------------------------------------------
__global__
void calcPartSum_kernel(float* d_parr_out, const int lenChunk, const int npol_physical, cufftComplex* d_parr_inp)
{
	int ichan = blockIdx.y;
	int ind = blockIdx.x * blockDim.x + threadIdx.x;
	if (ind < lenChunk)
	{
		float sum = 0;
		for (int i = 0; i < npol_physical; ++i)
		{
			sum += fnc_norm2(&d_parr_inp[(ichan * npol_physical + i) * lenChunk + ind]);
		}
		d_parr_out[ichan * lenChunk + ind] = sum;
	}
}
//------------------------------------------
__global__
void calcPowerMtrx_kernel(float* output, const int height, const int width, const int npol, cufftComplex* input)
{
	__shared__ float tile[TILE_DIM][TILE_DIM + 1]; // Shared memory tile
	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;
	int ichan = blockIdx.z;
	// Transpose data from global to shared memory
	if (x < width && y < height)
	{
		float sum = 0.;
		for (int i = 0; i < npol / 2; ++i)
		{
			sum += fnc_norm2(&input[(ichan * npol / 2 + i) * height * width + y * width + x]);
		}

		tile[threadIdx.y][threadIdx.x] = sum;
	}
	__syncthreads();

	// Calculate new indices for writing to output
	x = blockIdx.y * TILE_DIM + threadIdx.x;
	y = blockIdx.x * TILE_DIM + threadIdx.y;

	// Transpose data from shared to global memory
	if (x < height && y < width) {
		output[ichan * height * width + y * height + x] = tile[threadIdx.x][threadIdx.y];
	}
}
//------------------------------------------
__global__
void multiTransp_kernel(float* output, const int height, const int width, float* input)
{
	__shared__ float tile[TILE_DIM][TILE_DIM + 1]; // Shared memory tile

	int numchan = blockIdx.z;
	float* pntInp = &input[numchan * height * width];
	float* pntOut = &output[numchan * height * width];

	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;

	// Transpose data from global to shared memory
	if (x < width && y < height) {
		tile[threadIdx.y][threadIdx.x] = pntInp[y * width + x];
	}

	__syncthreads();

	// Calculate new indices for writing to output
	x = blockIdx.y * TILE_DIM + threadIdx.x;
	y = blockIdx.x * TILE_DIM + threadIdx.y;

	// Transpose data from shared to global memory
	if (x < height && y < width) {
		pntOut[y * height + x] = tile[threadIdx.x][threadIdx.y];
	}
}

//---------------------------------------------------------------

__global__ 
void normalize_and_clean(fdmt_type_* parrOut, float* d_arr, const int NRows, const int NCols
	,float *pmean, float *pstd, float* d_arrRowDisp, float *pmeanDisp, float *pstdDisp)
{
	__shared__ int sbad[1];
	unsigned int i = threadIdx.x;
	unsigned int irow = blockIdx.y;
	if (i >= NCols)
	{
		return;
	}
	if (fabs(d_arrRowDisp[irow] - *pmeanDisp) > 4. * (*pstdDisp))
	{
		sbad[0] = 1;
	}
	else
	{
		sbad[0] = 0;
	}
	//--------------------------------
	if (sbad[0] == 1)
	{
		while (i < NCols)
		{
			parrOut[irow * NCols + i] = 0;
			i += blockDim.x;
		}
	}
	else
	{
		while (i < NCols)
		{
			parrOut[irow * NCols + i] = (fdmt_type_)((d_arr[irow * NCols + i] - (*pmean) )/((*pstd )));
			i += blockDim.x;
		}
	}
	

}

//-------------------------------------------
void CChunk_gpu::preparations_and_memoryAllocations(CTelescopeHeader header
	, const float pulse_length
	, const float d_max
	, const float sigma_bound
	, const int length_sum_wnd
	, int* pLenChunk
	, cufftHandle* pplan0, cufftHandle* pplan1, CFdmtU* pfdmt, char** d_pparrInput, cufftComplex** ppcmparrRawSignalCur
	, void** ppAuxBuff_fdmt, fdmt_type_** d_parrfdmt_norm
	, cufftComplex** ppcarrTemp
	, cufftComplex** ppcarrCD_Out
	, cufftComplex** ppcarrBuff, char** ppInpOutBuffFdmt,  CChunk_gpu** ppChunk)
{
	//cudaError_t cudaStatus;
	//const float VAlFmin = header.m_centfreq - ((float)header.m_nchan) * header.m_chanBW / 2.0;
	//const float VAlFmax = header.m_centfreq + ((float)header.m_nchan) * header.m_chanBW / 2.0;
	//// 3.2 calculate standard len_sft and LenChunk    
	//const int len_sft = calc_len_sft(fabs(header.m_chanBW), pulse_length);
	//*pLenChunk = calcLenChunk_(header, len_sft, pulse_length, d_max);


	//// 3.3 cuFFT plans preparations

	//cufftCreate(pplan0);
	//checkCudaErrors(cufftPlan1d(pplan0, *pLenChunk, CUFFT_C2C, header.m_nchan * header.m_npol / 2));


	//
	//cufftCreate(pplan1);
	//checkCudaErrors(cufftPlan1d(pplan1, len_sft, CUFFT_C2C, (*pLenChunk) * header.m_nchan * header.m_npol / 2 / len_sft));
	//


	//// !3

	//// 4. memory allocation in GPU
	//// total number of downloding bytes to each file:
	//const long long QUantDownloadingBytesForChunk = (*pLenChunk) * header.m_nchan / 8 * header.m_nbits* header.m_npol;

	//const long long QUantBlockComplexNumbers = (*pLenChunk) * header.m_nchan * header.m_npol / 2;



	//checkCudaErrors(cudaMallocManaged((void**)d_pparrInput, QUantDownloadingBytesForChunk * sizeof(char)));


	//checkCudaErrors(cudaMalloc((void**)ppcmparrRawSignalCur, QUantBlockComplexNumbers * sizeof(cufftComplex)));
	//// 2!

	//

	//// 4.memory allocation for auxillary buffer for fdmt   
	//   // there is  quantity of real channels
	//const int NChan_fdmt_act = len_sft * header.m_nchan;
	//(*pfdmt) = CFdmtU(
	//	VAlFmin
	//	, VAlFmax
	//	, NChan_fdmt_act
	//	, (*pLenChunk) / len_sft
	//	, pulse_length
	//	, d_max
	//	, len_sft);

	//

	//size_t szBuff_fdmt = pfdmt->calcSizeAuxBuff_fdmt_();

	//checkCudaErrors(cudaMalloc(ppAuxBuff_fdmt, szBuff_fdmt));
	//// 4!
	//

	//// 3. memory allocation for fdmt_ones on GPU  ????
	//size_t szBuff_fdmt_output = pfdmt->calc_size_output();

	//checkCudaErrors(cudaMalloc((void**)d_parrfdmt_norm, szBuff_fdmt_output));
	////// 6. calculation fdmt ones
	//pfdmt->process_image(nullptr      // on-device input image
	//	, *ppAuxBuff_fdmt
	//	, *d_parrfdmt_norm	// OUTPUT image,
	//	, true);

	//// 3!

	//


	//// 5. memory allocation for the 3 auxillary cufftComplex  arrays on GPU	
	////cufftComplex* pffted_rowsignal = NULL; //1	



	//checkCudaErrors(cudaMalloc((void**)ppcarrTemp, QUantBlockComplexNumbers * sizeof(cufftComplex)));

	//checkCudaErrors(cudaMalloc((void**)ppcarrCD_Out, QUantBlockComplexNumbers * sizeof(cufftComplex)));

	//checkCudaErrors(cudaMalloc((void**)ppcarrBuff, QUantBlockComplexNumbers * sizeof(cufftComplex)));
	//// !5
	//
	//// 5. memory allocation for the 2 auxillary arrays on GPU for input and output of FDMT	
	//size_t szInpOut_fdmt = pfdmt->calc_size_output() + pfdmt->calc_size_input();

	//checkCudaErrors(cudaMalloc((void**)ppInpOutBuffFdmt, szInpOut_fdmt));

	//// 5!
	//
	//// !4	
	//**ppChunk = CChunk_gpu(
	//	VAlFmin
	//	, VAlFmax
	//	, header.m_npol
	//	, header.m_nchan
	//	, (*pLenChunk)
	//	, len_sft
	//	, 0
	//	, 0
	//	, header.m_nbits
	//	, d_max
	//	, sigma_bound
	//	, length_sum_wnd
	//	, *pfdmt
	//	, pulse_length
	//);
	//
}
//-----------------------------------------------------------------------------------------
void windowization(float* d_fdmt_normalized, const int Rows, const int Cols, const int width, float* parrImage)
{
	for (int i = 0; i < Rows; ++i)
	{
		for (int j = 0; j < Cols; ++j)
		{
			
			float sum = 0.;
			for (int k = 0; k < width; ++k)
			{
				if ((j + k) < Cols)
				{
					sum += d_fdmt_normalized[i * Cols + j + k];
				}
				else
				{
					sum = 0.;
					break;
				}
				
			}
			parrImage[i * Cols + j] = sum / sqrt((float)width);
		}
	}
}
//----------------------------------------------------
__global__
void fdmt_normalization(fdmt_type_* d_arr, fdmt_type_* d_norm, const int len, float* d_pOutArray)
{

	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= len)
	{
		return;
	}
	d_pOutArray[idx] = ((float)d_arr[idx]) / sqrtf(((float)d_norm[idx]) + 1.0E-8);

}








