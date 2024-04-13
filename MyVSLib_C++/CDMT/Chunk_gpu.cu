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


	CChunk_gpu::CChunk_gpu() :CChunkB()
	{

	}
	//-----------------------------------------------------------

	CChunk_gpu::CChunk_gpu(const  CChunk_gpu& R) :CChunkB(R)
	{
	}
	//-------------------------------------------------------------------

	CChunk_gpu& CChunk_gpu::operator=(const CChunk_gpu& R)
	{
		if (this == &R)
		{
			return *this;
		}
		CChunkB:: operator= (R);
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
		, const double d_max
		, const double d_min
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
	}

	//
	//
	////---------------------------------------------------
	bool CChunk_gpu::process(void* pcmparrRawSignalCur
		, std::vector<COutChunkHeader>* pvctSuccessHeaders, std::vector<std::vector<float>>* pvecImg)
	{

		return true;
	}

	bool CChunk_gpu::try0()
	{
		return true;
	}
	//-----------------------------------------------------------------

//-----------------------------------------
bool CChunk_gpu::detailedChunkProcessing(
	  const COutChunkHeader outChunkHeader
	, cufftHandle plan0
	, cufftHandle plan1
	, cufftComplex* pcmparrRawSignalCur
	, fdmt_type_* d_arrfdmt_norm
	, void* pAuxBuff_fdmt
	, cufftComplex* pcarrTemp
	, cufftComplex* pcarrCD_Out
	, cufftComplex* pcarrBuff
	, char* pInpOutBuffFdmt	
	, CFragment* pFRg)
{
	// 1. installation of pointers	for pAuxBuff_the_rest
	fdmt_type_* d_parr_fdmt_inp = (fdmt_type_*)pInpOutBuffFdmt; //4	
	fdmt_type_* d_parr_fdmt_out = (fdmt_type_*)pInpOutBuffFdmt + m_Fdmt.m_nchan * m_Fdmt.m_cols;
	//// !1

	 
	auto start = std::chrono::high_resolution_clock::now();


	// 2. create FFT	

	checkCudaErrors(cufftExecC2C(plan0, pcmparrRawSignalCur, pcmparrRawSignalCur, CUFFT_FORWARD));

	// !2

	// 3.

	

	/*int n_coherent = calc_n_coherent(m_Fmin, m_Fmax, m_nchan, m_len_sft, m_Fdmt.m_imaxDt);
	cout << "n_coherent =  " << n_coherent << std::endl;*/
	// !3

	//// 4. main loop
	
	bool breturn = false;
	


	

	fncCD_Multiband_gpu(pcarrCD_Out, pcmparrRawSignalCur
		, DISPERSION_CONSTANT * outChunkHeader.m_coherentDedisp, plan0, pcarrTemp);

	

	//std::vector<std::complex<float>> data0(m_nbin * m_nchan * m_npol / 2, 0);
	//cudaMemcpy(data0.data(), pcarrCD_Out, m_nbin * m_nchan * m_npol / 2 * sizeof(std::complex<float>),
	//	cudaMemcpyDeviceToHost);
	//cudaDeviceSynchronize();

	cufftResult result = cufftExecC2C(plan1, pcarrCD_Out, pcarrTemp, CUFFT_FORWARD);
	cudaDeviceSynchronize();

	/*std::vector<std::complex<float>> data(m_nbin * m_nchan * m_npol / 2, 0);
	cudaMemcpy(data.data(), pcarrTemp, m_nbin * m_nchan * m_npol / 2 * sizeof(std::complex<float>),
		cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();*/

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	iFFT_time += duration.count();
	//

	calc_fdmt_inp(d_parr_fdmt_inp, pcarrTemp, (float*)pcarrCD_Out);
	//
	/*std::vector<float> data(m_nbin * m_nchan * m_npol / 2, 0);
	cudaMemcpy(data.data(), d_parr_fdmt_inp, m_nbin * m_nchan * m_npol / 2 * sizeof(float),
		cudaMemcpyDeviceToHost);
	std::array<long unsigned, 2> leshape129{ m_len_sft * m_nchan, m_nbin / m_len_sft };
	npy::SaveArrayAsNumpy("fdmt_inp.npy", false, leshape129.size(), leshape129.data(), data);*/


	start = std::chrono::high_resolution_clock::now();

	end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	iFdmt_time += duration.count();

	m_Fdmt.process_image(d_parr_fdmt_inp	, d_parr_fdmt_out, false);

	float* d_fdmt_normalized = (float*)malloc(m_nbin * m_nchan * sizeof(float));
	cudaMallocManaged((void**)&d_fdmt_normalized, m_nbin * m_nchan * sizeof(float));
	const int Rows = m_Fdmt.m_imaxDt;
	const int Cols = m_Fdmt.m_cols;
	int theads = 1024;
	int blocks = (m_nbin + theads - 1) / theads;
	fdmt_normalization << < blocks, theads >> > (d_parr_fdmt_out, d_arrfdmt_norm, m_nbin * m_nchan, d_fdmt_normalized);
	cudaDeviceSynchronize();

	float* h_parrImage = (float*)malloc(Rows * Cols * sizeof(float));
	windowization(d_fdmt_normalized, Rows, Cols, outChunkHeader.m_wnd_width, h_parrImage);

	const int IDim = (Rows < Cols) ? Rows : Cols;

	float* parrFragment = (float*)malloc(IDim * IDim * sizeof(float));
	int iRowBegin = -1, iColBegin = -1;

	cutQuadraticFragment(parrFragment, h_parrImage, &iRowBegin, &iColBegin
		, Rows, Cols, outChunkHeader.m_nSucessRow, outChunkHeader.m_nSucessCol);

	CFragment frg(parrFragment, IDim, iRowBegin, iColBegin, outChunkHeader.m_wnd_width, outChunkHeader.m_SNR, outChunkHeader.m_coherentDedisp);
	*pFRg = frg;
	cudaFree(d_fdmt_normalized);

	free(h_parrImage);
	free(parrFragment);
	
	return true;
}
//---------------------------------------------------
bool CChunk_gpu::fncChunkProcessing_gpu(cufftComplex* pcmparrRawSignalCur
	, void* pAuxBuff_fdmt
	, cufftComplex* pcarrTemp
	, cufftComplex* pcarrCD_Out
	, cufftComplex* pcarrBuff
	, char* pInpOutBuffFdmt
	, fdmt_type_* d_arrfdmt_norm
	, cufftHandle plan0, cufftHandle plan1	
	, std::vector<COutChunkHeader>* pvctSuccessHeaders)
{
	cudaError_t cudaStatus;
	
	// 1. installation of pointers	for pAuxBuff_the_rest
	fdmt_type_* d_parr_fdmt_inp = (fdmt_type_*)pInpOutBuffFdmt; //4	
	fdmt_type_* d_parr_fdmt_out =  (fdmt_type_ *)pInpOutBuffFdmt + m_Fdmt.m_nchan * m_Fdmt.m_cols;
	//// !1

	/* std::vector<std::complex<float>> data2(m_nbin, 0);
	cudaMemcpy(data2.data(), pcmparrRawSignalCur, m_nbin * sizeof(std::complex<float>),
		cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();*/
	//std::array<long unsigned, 1> leshape127{ LEnChunk };
	//npy::SaveArrayAsNumpy("ffted.npy", false, leshape127.size(), leshape127.data(), data);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus));
		// Handle the error appropriately
	}
	auto start = std::chrono::high_resolution_clock::now();
	

	// 2. create FFT	
	
	checkCudaErrors(cufftExecC2C(plan0, pcmparrRawSignalCur, pcmparrRawSignalCur, CUFFT_FORWARD));

	// !2

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	iFFT_time += duration.count();

	/*std::vector<std::complex<float>> data(m_nbin * m_nchan * m_npol/2, 0);
	cudaMemcpy(data.data(), pcmparrRawSignalCur, m_nbin * m_nchan * m_npol/2 * sizeof(std::complex<float>),
		cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();*/


	// 3.
	
	
	
	//int n_coherent = calc_n_coherent(m_Fdmt.m_Fmin, m_Fdmt.m_Fmax, m_Fdmt.m_nchan/ m_len_sft, m_d_max, m_pulse_length);
	
	//cout << "n_coherent =  " << n_coherent << std::endl;
	// !3
	

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus));
		// Handle the error appropriately
	}

	

	structOutDetection* pstructOutCur = 0;
	checkCudaErrors(cudaMallocManaged((void**)&pstructOutCur, sizeof(structOutDetection)));
	cudaDeviceSynchronize();
	pstructOutCur->snr = 1. - FLT_MAX;
	pstructOutCur->icol = -1;
	pstructOutCur->irow = -1;
	pstructOutCur->iwidth = -1;


	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus));
		// Handle the error appropriately
	}
	//// 4. main loop
	float coherent_d = -1.;
	bool breturn = false;
	//for (int iouter_d =31; iouter_d < 32; ++iouter_d)
	for (int iouter_d = 0; iouter_d < m_ncoherent; ++iouter_d)
	{
		/*if (iouter_d < 30.0)
		{
			continue;
		}*/
		double valcur_coherent_d = ((double)iouter_d) * ((double)m_d_max / ((double)m_ncoherent));
		cout << "Chunk=  " << m_Chunk_id <<  "; iter= " << iouter_d << "; coherent_d = " << valcur_coherent_d << endl;

		/*std::vector<std::complex<float>> data(m_nbin * m_nchan * m_npol /2, 0);
		cudaMemcpy(data.data(), pcmparrRawSignalCur, m_nbin * m_nchan * m_npol / 2 * sizeof(std::complex<float>),
		cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();*/
		
		// fdmt input matrix computation
		/*calcFDMT_Out_gpu(d_parr_fdmt_out, pcmparrRawSignalCur, pcarrCD_Out
			, pcarrTemp, d_parr_fdmt_inp
			, DISPERSION_CONSTANT * valcur_coherent_d
			, pAuxBuff_fdmt, plan0, plan1, pcarrBuff);*/
		// !
		///*float* parr_fdmt_out = (float*)malloc(m_nbin * m_nchan* sizeof(float));
		//cudaMemcpy(parr_fdmt_out, d_parr_fdmt_out, m_nbin * m_nchan * sizeof(float), cudaMemcpyDeviceToHost);
		//float valmax = -0., valmin = 0.;
		//unsigned int iargmax = -1, iargmin = -1;
		//findMaxMinOfArray(parr_fdmt_out, m_nbin * m_nchan, &valmax, &valmin
		//	, &iargmax, &iargmin);
		//float* arrfdmt_norm = (float*)malloc(m_nbin * m_nchan * sizeof(float));
		//cudaMemcpy(arrfdmt_norm, d_arrfdmt_norm, m_nbin * m_nchan * sizeof(float), cudaMemcpyDeviceToHost);
		//float valmax1 = -0., valmin1 = 0.;
		//unsigned int iargmax1 = -1, iargmin1 = -1;
		//findMaxMinOfArray(arrfdmt_norm, m_nbin * m_nchan, &valmax1, &valmin1
		//	, &iargmax1, &iargmin1);
		//free(parr_fdmt_out);
		//free(arrfdmt_norm);*/

		fncCD_Multiband_gpu(pcarrCD_Out, pcmparrRawSignalCur
			, DISPERSION_CONSTANT * valcur_coherent_d, plan0, pcarrTemp);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus));
			// Handle the error appropriately
		}

		auto start = std::chrono::high_resolution_clock::now();

		/*std::vector<std::complex<float>> data0(m_nbin * m_nchan * m_npol / 2, 0);
		cudaMemcpy(data0.data(), pcarrCD_Out, m_nbin * m_nchan * m_npol / 2 * sizeof(std::complex<float>),
			cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();*/

		cufftResult result = cufftExecC2C(plan1, pcarrCD_Out, pcarrTemp, CUFFT_FORWARD);
		cudaDeviceSynchronize();

		/*std::vector<std::complex<float>> data(m_nbin * m_nchan * m_npol / 2, 0);
		cudaMemcpy(data.data(), pcarrTemp, m_nbin * m_nchan * m_npol / 2 * sizeof(std::complex<float>),
			cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();*/

		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
		iFFT_time += duration.count();
		//
		
		calc_fdmt_inp(d_parr_fdmt_inp, pcarrTemp, (float*)pcarrCD_Out);
		//
		/*std::vector<float> data4(m_nbin * m_nchan * m_npol / 2, 0);
		cudaMemcpy(data4.data(), d_parr_fdmt_inp, m_nbin * m_nchan * m_npol / 2 * sizeof(float),
			cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();*/
		/*std::array<long unsigned, 2> leshape129{ m_len_sft * m_nchan, m_nbin / m_len_sft };
		npy::SaveArrayAsNumpy("fdmt_inp.npy", false, leshape129.size(), leshape129.data(), data);*/

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus));
			// Handle the error appropriately
		}
		start = std::chrono::high_resolution_clock::now();
		
		end = std::chrono::high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
		iFdmt_time += duration.count();

		m_Fdmt.process_image(d_parr_fdmt_inp	, d_parr_fdmt_out, false);
		/*std::vector<float> data5(m_Fdmt.m_imaxDt * m_Fdmt.m_cols, 0);
		cudaMemcpy(data5.data(), d_parr_fdmt_out, m_Fdmt.m_imaxDt * m_Fdmt.m_cols * sizeof(float),
			cudaMemcpyDeviceToHost);*/
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus));
			// Handle the error appropriately
		}
		
		//
		const int Rows = m_Fdmt.m_imaxDt;
		const int Cols = m_Fdmt.m_cols;
		const dim3 Totalsize(1024, 1, 1);
		const dim3 gridSize((Cols + Totalsize.x - 1) / Totalsize.x, Rows, 1);
		
		float* d_pAuxArray = (float*)d_parr_fdmt_inp;
		int* d_pAuxNumArray = (int*)(d_pAuxArray + gridSize.x * gridSize.y);
		int* d_pWidthArray = d_pAuxNumArray + gridSize.x * gridSize.y;
		/*float* d_pAuxArray    = (float*)pcarrTemp;
		int*   d_pAuxNumArray = (int*)pcarrCD_Out;
		int*   d_pWidthArray  = (int*)pcarrBuff;*/
		detect_signal_gpu(d_parr_fdmt_out, d_arrfdmt_norm, Rows
			, Cols, m_length_sum_wnd, gridSize, Totalsize
			, d_pAuxArray, d_pAuxNumArray, d_pWidthArray, pstructOutCur);
		cudaDeviceSynchronize();

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus));
			// Handle the error appropriately
		}
		if ((*pstructOutCur).snr >= m_sigma_bound)
		{

			COutChunkHeader head(Rows
				, Cols
				, (*pstructOutCur).irow
				, (*pstructOutCur).icol
				, (*pstructOutCur).iwidth
				, (*pstructOutCur).snr
				, valcur_coherent_d
				, m_Block_id
				, m_Chunk_id
			);
			pvctSuccessHeaders->push_back(head);

			std::cout << "SNR = " << (*pstructOutCur).snr << "ROW ARGMAX = " << (*pstructOutCur).irow << "COLUMN ARGMAX = " << (*pstructOutCur).icol << endl;
			
			int frequency = 1500; // Frequency in hertz
			int duration = 500;   // Duration in milliseconds
			//emitSound(frequency, duration / 4);
			//emitSound(frequency + 500, duration / 2);
			d_pAuxArray = NULL;
			d_pAuxNumArray = NULL;

			d_pWidthArray = NULL;

			breturn = true;
		}
		//
		///*std::vector<float> data(LEnChunk, 0);
		//cudaMemcpy(data.data(), parr_fdmt_out, LEnChunk * sizeof(float),
		//	cudaMemcpyDeviceToHost);
		//cudaDeviceSynchronize();*/
	}
	cudaFree(pstructOutCur);
	return breturn;
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
//		double valcur_coherent_d = ((double)iouter_d) * ((double)m_d_max / ((double)n_coherent));
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

//-------------------------------------------------------------------
void CChunk_gpu::fncCD_Multiband_gpu(cufftComplex* pcarrCD_Out, cufftComplex* pcarrffted_rowsignal
	, const  double VAl_practicalD, cufftHandle  plan, cufftComplex* pAuxBuff)
{	
	 const dim3 Totalsize = dim3(1024, 1, 1);
	 const dim3 gridSize = dim3((m_nbin + Totalsize.x - 1) / Totalsize.x,  m_nchan,1);	
	 kernel_ElementWiseMult << < gridSize, Totalsize >> >
		 (pAuxBuff,pcarrffted_rowsignal, m_nbin, m_npol/2, VAl_practicalD, m_Fmin
		 	, m_Fmax);
	 
	 cudaDeviceSynchronize();

	/*std::vector<std::complex<float>> data(m_nbin * m_nchan * m_npol / 2, 0);
	cudaMemcpy(data.data(), pAuxBuff, m_nbin * m_nchan * m_npol / 2 * sizeof(std::complex<float>),
		cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();*/

	auto start = std::chrono::high_resolution_clock::now();
	checkCudaErrors(cufftExecC2C(plan, pAuxBuff, pcarrCD_Out, CUFFT_INVERSE));

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	iFFT_time += duration.count();

	/*std::vector<std::complex<float>> data1(m_nbin * m_nchan * m_npol / 2, 0);
	cudaMemcpy(data1.data(), pcarrCD_Out, m_nbin * m_nchan * m_npol / 2 * sizeof(std::complex<float>),
		cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();*/
	
	scaling_kernel << <1, 1024, 0 >> > (pcarrCD_Out, m_nbin * m_npol* m_nchan/2, 1.f / ((float)m_nbin));
	cudaDeviceSynchronize();
}

//-------------------------------------------------------------
 // calculation:
 // H = np.e**(-(2*np.pi*complex(0,1) * practicalD /(f_min + f) + 2*np.pi*complex(0,1) * practicalD*f /(f_max**2)))
 // np.fft.fft(raw_signal) * H 
// gridDim.z - num channels
// gridDim.y - num polarizations in physical sense
__global__ void kernel_ElementWiseMult(cufftComplex* pAuxBuff, cufftComplex* pcarrffted_rowsignal
	, const unsigned int LEnChunk, const unsigned int n_pol_phys
	, const  double VAl_practicalD, const double Fmin
	, const double Fmax)
{
	__shared__ double arrf[2];

	double chanBW = (Fmax - Fmin) / gridDim.y;
	arrf[0] = Fmin + chanBW * blockIdx.y;
	arrf[1] = arrf[0] + chanBW;	

	const int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j >= LEnChunk)
	{
		return;
	}
	double step = (arrf[1] - arrf[0]) / ((double)LEnChunk);
	//double t = VAl_practicalD * (1. / (arrf[0] + step * (double)j) + 1. / (arrf[1]) * (step * (double)j / arrf[1]));
	double t = VAl_practicalD * (1. / (arrf[0] + step * (double)j) + 1. / (arrf[1]) * (step * (double)j / arrf[1]))
		+ (VAl_practicalD / arrf[0] + VAl_practicalD * (arrf[0] - Fmin) / (Fmax * Fmax));
	/*double step = (Fmax - Fmin) / ((double)LEnChunk);
	double t = VAl_practicalD * (1. / (Fmin + step * (double)j) + 1. / (Fmax) * (step * (double)j / Fmax));	*/

	double val_prD_int = 0, val_prD_frac = 0;
	double t4 = -modf(t, &val_prD_int) * 2.0;

	double val_x = 0.;
	double val_y = 0.;
	sincospi(t4, &val_y, &val_x);
	unsigned int nelem0 = LEnChunk * blockIdx.y * n_pol_phys + j;
	pAuxBuff[nelem0].x = (float)(val_x * pcarrffted_rowsignal[nelem0].x - val_y * pcarrffted_rowsignal[nelem0].y); // Real part
	pAuxBuff[nelem0].y = (float)(val_x * pcarrffted_rowsignal[nelem0].y + val_y * pcarrffted_rowsignal[nelem0].x); // Imaginary part
	if (n_pol_phys == 2)
	{
		nelem0 += LEnChunk;
		pAuxBuff[nelem0].x = (float)(val_x * pcarrffted_rowsignal[nelem0].x - val_y * pcarrffted_rowsignal[nelem0].y); // Real part
		pAuxBuff[nelem0].y = (float)(val_x * pcarrffted_rowsignal[nelem0].y + val_y * pcarrffted_rowsignal[nelem0].x); // Imaginary par
	}
}

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








