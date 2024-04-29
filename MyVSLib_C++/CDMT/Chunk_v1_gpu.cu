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
		
	}
	//-----------------------------------------------------------
	CChunk_v1_gpu::CChunk_v1_gpu() :CChunk_gpu()
	{		
	}
	//-----------------------------------------------------------

	CChunk_v1_gpu::CChunk_v1_gpu(const  CChunk_v1_gpu& R) :CChunk_gpu(R)
	{
	}
	//-------------------------------------------------------------------

	CChunk_v1_gpu& CChunk_v1_gpu::operator=(const CChunk_v1_gpu& R)
	{
		if (this == &R)
		{
			return *this;
		}
		CChunk_gpu:: operator= (R);

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
	kernel_create_arr_dc1 << < gridSize2, block_Size2 >> > (m_pd_arr_dc, m_pd_arrcoh_dm, d_parr_freqs_chan, d_parr_bin_freqs, d_parr_taper, ndm, m_nchan, m_len_sft, mbin);
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
//__global__
//void kernel_create_arr_dc(cufftComplex* parr_dc, double* parrcoh_dm, double* parr_freqs_chan, double* parr_bin_freqs, double* parr_taper
//	, int ndm, int nchan, int len_sft, int mbin)
//{
//	//__shared__  double temp0[1];
//	//__shared__  int i0[1];
//
//	int ibin = blockIdx.x * blockDim.x + threadIdx.x;
//	if (ibin >= mbin)
//	{
//		return;
//	}
//
//	int num1 = blockIdx.y * blockDim.y + threadIdx.y;
//	if (num1 >= nchan * len_sft)
//	{
//		return;
//	}
//
//	int ichan = num1 / len_sft;
//	int isft = num1 % len_sft;
//
//	int idm = blockIdx.z * blockDim.z + threadIdx.z;
//	if (idm >= ndm)
//	{
//		return;
//	}
//
//	float temp0 = parr_freqs_chan[ichan * len_sft + isft];
//	int i0 = idm * nchan * len_sft * mbin + ichan * len_sft * mbin + isft * mbin;
//
//	double temp1 = parr_bin_freqs[ibin] / temp0;
//	double phase_delay = (parrcoh_dm[idm] * temp1 * temp1 / (temp0 + parr_bin_freqs[ibin]) * 4.148808e9);
//	double val_prD_int = 0;
//	double t = -modf(phase_delay, &val_prD_int) * 2.0;
//	double val_x = 0.0, val_y = 0.;
//
//	sincospi(t, &val_y, &val_x);
//	parr_dc[i0 + ibin].x = float(val_x * parr_taper[ibin]);
//	parr_dc[i0 + ibin].y = float(val_y * parr_taper[ibin]);
//}


//-------------------------------------------------------------------------------------------
__global__
void kernel_create_arr_dc1  (cufftComplex* parr_dc, double * parrcoh_dm,double* parr_freqs_chan, double* parr_bin_freqs, double *parr_taper
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
	
	
	parr_dc[i1 * nbin + ibin1].x = float(val_x * parr_taper[ibin]/*/((double)nbin)*/);
	parr_dc[i1 * nbin + ibin1].y = float(val_y * parr_taper[ibin] /*/ ((double)nbin)*/);
}

//------------------------------------------------------------------------------------------

	bool CChunk_v1_gpu::process(void* pcmparrRawSignalCur
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
		
		cudaStatus0 = cudaGetLastError();
		if (cudaStatus0 != cudaSuccess) {
			fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus0));
			return false;
		}
		/*cufftComplex* pcmparrRawSignalRolled = NULL;
		checkCudaErrors(cudaMalloc((void**) & pcmparrRawSignalRolled, m_nfft * m_nchan * m_npol / 2 * m_nbin * sizeof(cufftComplex)));
		*/
		
		//3!
		
		for (int idm = 0; idm < m_coh_dm_Vector.size(); ++idm)
		{
			dim3 threadsPerBlock(1024, 1, 1);
			dim3 blocksPerGrid((m_nchan * m_nbin + threadsPerBlock.x - 1) / threadsPerBlock.x, m_nfft, m_npol/2);
			element_wise_cufftComplex_mult_kernel<<< blocksPerGrid, threadsPerBlock>>>
				(m_pdcmpbuff_ewmulted,(cufftComplex*)pcmparrRawSignalCur, &m_pd_arr_dc[m_nchan * m_nbin * idm], m_npol/2,m_nfft, m_nchan * m_nbin);
			cudaDeviceSynchronize();					
			// result stored in m_pdcmpbuff_ewmulted
			cudaStatus0 = cudaGetLastError();
			if (cudaStatus0 != cudaSuccess) {
				fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus0));
				return false;
			}
			checkCudaErrors(cufftExecC2C(m_fftPlanInverse, m_pdcmpbuff_ewmulted, m_pdcmpbuff_ewmulted, CUFFT_INVERSE));


			int threads1 = 1024;
			int blocks1 = (m_nfft * m_nchan * m_npol / 2 * m_nbin + threads1 - 1) / threads1;	

			roll_rows_normalize_sum_kernel << <blocks1, threads1 >> > (m_pdbuff_rolled, m_pdcmpbuff_ewmulted, m_npol, m_nfft * m_nchan , m_nbin, m_nbin / 2);
			cudaDeviceSynchronize();
			// result stored in m_pdbuff_rolled		
			cudaStatus0 = cudaGetLastError();
			if (cudaStatus0 != cudaSuccess) {
				fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus0));
				return false;
			}

			int noverlap_per_channel = get_noverlap_per_channel();
			int mbin_adjusted = get_mbin_adjusted();
			// result stored in m_pdbuff_rolled	
			float* fbuf = (float*) m_pdcmpbuff_ewmulted;
					
			
			dim3 threads_per_block1(1024, 1, 1);
			dim3 blocks_per_grid1(( mbin_adjusted + threads_per_block1.x - 1) / threads_per_block1.x, m_nchan * m_len_sft, m_nfft);
			
			transpose_unpadd_intensity << < blocks_per_grid1, threads_per_block1 >> > 
				(fbuf, m_pdbuff_rolled, m_nfft, noverlap_per_channel
				, mbin_adjusted, m_nchan, m_len_sft, mbin);
			cudaDeviceSynchronize();
			// result stored in m_pdcmpbuff_ewmulted(=fbuf).  m_pdbuff_rolled	 is free
			cudaStatus0 = cudaGetLastError();
			if (cudaStatus0 != cudaSuccess) {
				fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus0));
				return false;
			}

			float* parr_wfall = m_pdbuff_rolled;
			
			//
			dim3 threadsPerChunk(TILE_DIM, TILE_DIM, 1);
			dim3 chunkPerGrid((m_nchan* m_len_sft + TILE_DIM - 1) / TILE_DIM, ( msamp + TILE_DIM - 1) / TILE_DIM, 1);
			

			transpose_ << <  chunkPerGrid, threadsPerChunk >> > (parr_wfall, fbuf, m_nchan * m_len_sft, msamp);
			cudaDeviceSynchronize();
			cudaStatus0 = cudaGetLastError();
			// result stored in m_pdbuff_rolled(=parr_wfall).  m_pdcmpbuff_ewmulted	 is free
			
			float* parr_wfall_disp = (float*)m_pdcmpbuff_ewmulted;
			double val_tsamp_wfall = m_len_sft * m_tsamp;
			double val_dm = m_coh_dm_Vector[idm];
			double f0 = ((double)m_Fmax - (double)m_Fmin) / (m_nchan * m_len_sft);
			dim3 threadsPerblock1 (1024,1,1);
			dim3 blocksPerGrid1  ((msamp + threadsPerblock1.x - 1) / threadsPerblock1.x, m_nchan * m_len_sft, 1);
			dedisperse << <  blocksPerGrid1, threadsPerblock1 >> > (parr_wfall_disp, parr_wfall, val_dm
				, (double)m_Fmin, (double)m_Fmax,  val_tsamp_wfall, f0, msamp);
			cudaDeviceSynchronize();
						
			
			m_Fdmt.process_image(parr_wfall_disp, &m_pdoutImg[idm * msamp * m_len_sft], false);

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
			cudaMemcpy(pvecImg_temp->at(i).data(), &m_pdoutImg[i * msamp * m_len_sft], msamp* m_len_sft * sizeof(float), cudaMemcpyDeviceToHost);
		}
		cudaStatus0 = cudaGetLastError();
		if (cudaStatus0 != cudaSuccess) {
			fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus0));
			return false;
		}
		return true;
	}

//-----------------------------------------------------------------------------

	//__global__	void  transpose_unpadd_kernel(cufftComplex* fbuf, cufftComplex* arin, int nfft, int noverlap_per_channel
	//	, int mbin_adjusted, const int nsub, const int nchan, int mbin)
	//{
	//	int  ibin = blockIdx.x * blockDim.x + threadIdx.x;
	//	if (!(ibin < mbin_adjusted))
	//	{
	//		return;
	//	}
	//	int ipol = blockIdx.z / nfft;
	//	int ifft = blockIdx.z % nfft;
	//	int isub = blockIdx.y / nchan;
	//	int ichan = blockIdx.y % nchan;
	//	int ibin_adjusted = ibin + noverlap_per_channel;
	//	int isamp = ibin + mbin_adjusted * ifft;
	//	int msamp = mbin_adjusted * nfft;
	//	// Select bins from valid region and reverse the frequency axis		
	//   // printf("ipol = %i   ifft =  %i\n", ipol, ifft);
	//	int iinp = ipol * nfft * nsub * nchan * mbin + ifft * nsub * nchan * mbin + (nsub - isub - 1) * nchan * mbin + ichan * mbin + ibin_adjusted;
	//	int iout = ipol * msamp * nsub * nchan + isamp * nsub * nchan + isub * nchan + nchan - ichan - 1;
	//	// Select bins from valid region and reverse the frequency axis		

	//	fbuf[iout].x = arin[iinp].x;
	//	fbuf[iout].y = arin[iinp].y;
	//}


__global__ 	void  transpose_unpadd_intensity(float* fbuf, float* arin, int nfft, int noverlap_per_channel
	, int mbin_adjusted,  const int nsub, const int nchan, int mbin)
{
	int  ibin = blockIdx.x * blockDim.x + threadIdx.x;
	if (!(ibin < mbin_adjusted))
	{
		return;
	}	
	int ifft = blockIdx.z ;
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
	fbuf[iout] = arin[iinp];
}


//---------------------------------------------------------------------------
__global__ void roll_rows_normalize_sum_kernel(float* arr_rez, cufftComplex* arr, const int npol, const int rows
	, const  int cols, const  int shift)
{

	int idx0 = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx0 >= cols)
	{
		return;
	}
	int ind_new = blockIdx.y * cols + (idx0 + shift) % cols;
	int ind = blockIdx.y * cols + idx0;
	arr_rez[ind_new] = 0.0f;
	for (int ipol = 0; ipol < npol / 2; ++ipol)
	{
		int ind_cur = ind + ipol * rows * cols;
		double x = (double)arr[ind_cur].x;
		double y = (double)arr[ind_cur].y;
		arr_rez[ind_new] += float((x * x + y * y) / cols / cols);
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








