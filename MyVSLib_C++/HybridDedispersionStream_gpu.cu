#include "HybridDedispersionStream_gpu.cuh"
#include "CFdmtG.cuh"
#include "StreamParams.h"
#include "Constants.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <array>
#include <vector>
#include "npy.hpp"

#include <device_launch_parameters.h>
#include <cmath>

#include <chrono>
#include <math_functions.h>

//#include <cufft.h>
//#include <thrust/complex.h>
//#include <complex>

#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define TILE_DIM 32

using namespace std;

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



//#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL( call )                                                                                           \
    {                                                                                                                  \
        auto status = static_cast<cudaError_t>( call );                                                                \
        if ( status != cudaSuccess )                                                                                   \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "                                        \
                     "with "                                                                                           \
                     "%s (%d).\n",                                                                                     \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     cudaGetErrorString( status ),                                                                     \
                     status );                                                                                         \
    }
//#endif  // CUDA_RT_CALL

// cufft API error chekcing
//#ifndef CUFFT_CALL
#define CUFFT_CALL( call )                                                                                             \
    {                                                                                                                  \
        auto status = static_cast<cufftResult>( call );                                                                \
        if ( status != CUFFT_SUCCESS )                                                                                 \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUFFT call \"%s\" in line %d of file %s failed "                                          \
                     "with "                                                                                           \
                     "code (%d).\n",                                                                                   \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     status );                                                                                         \
    }
//#endif  // CUFFT_CALL
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

inline void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(result), cudaGetErrorString(result), func);
		exit(EXIT_FAILURE);
	}
}
//---------------------------------------------------------------------------------
int fncHybridDedispersionStream_gpu( int* piarrNumSuccessfulChunks, float* parrCoherent_d
	, int& quantOfSuccessfulChunks, CStreamParams* pStreamPars)
{
	cudaError_t cudaStatus;
	// 1. memory allocation for current chunk
	const int NumChunks = ((pStreamPars->m_numEnd - pStreamPars->m_numBegin) + pStreamPars->m_lenChunk - 1) / pStreamPars->m_lenChunk;
	cufftComplex* pcmparrRawSignalCur = NULL;
	cudaStatus = cudaMallocManaged((void**)&pcmparrRawSignalCur, pStreamPars->m_lenChunk * sizeof(cufftComplex));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "fncHybridDedispersionStream_gpu. cudaMallocManaged failed!");
		return 1;
	}	
	// 1!

	// 2. memory allocation for fdmt_ones on GPU
	fdmt_type_* d_arrfdmt_norm = 0;
	cudaStatus = cudaMallocManaged((void**)&d_arrfdmt_norm, pStreamPars->m_lenChunk * sizeof(fdmt_type_));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "fncHybridDedispersionStream_gpu. cudaMallocManaged failed!");
		return 1;
	}
	
	// 2!

	// 3.memory allocation for auxillary buffer for fdmt
	const int  IDeltaT = calc_IDeltaT(pStreamPars->m_n_p, pStreamPars->m_f_min, pStreamPars->m_f_max, pStreamPars->m_IMaxDT);
	size_t szBuff_fdmt = calcSizeAuxBuff_fdmt(pStreamPars->m_n_p, pStreamPars->m_lenChunk / pStreamPars->m_n_p
		, pStreamPars->m_f_min, pStreamPars->m_f_max, pStreamPars->m_IMaxDT);
	void* pAuxBuff_fdmt = 0;
	cudaStatus = cudaMalloc(&pAuxBuff_fdmt, szBuff_fdmt);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "fncHybridDedispersionStream_gpu. cudaMallocManaged for pAuxBuff_fdmt failed!");
		return 1;
	}
	// 3!

	// 4. memory allocation for the 4 auxillary cufftComplex  arrays on GPU	
	cufftComplex* pffted_rowsignal = NULL; //1	
	cufftComplex* pcarrTemp = NULL; //2	
	cufftComplex* pcarrCD_Out = NULL;//3
	cufftComplex* pcarrBuff = NULL;//3
	if (1 == malloc_for_4_complex_arrays(&pffted_rowsignal, &pcarrTemp, &pcarrCD_Out, &pcarrBuff, pStreamPars->m_lenChunk))
	{
		return 1;
	}		
	// !4

	// 5. memory allocation for the 2 auxillary float  arrays on GPU	
	fdmt_type_ * pAuxBuff_flt =  NULL;
	cudaStatus = cudaMalloc((void**)&pAuxBuff_flt, 2 * (pStreamPars->m_lenChunk) * sizeof(fdmt_type_ ));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "pffted_rowsignal. cudaMalloc failed!");
		return 1;
	}
	// 5!
	
	// 6. calculation fdmt ones
	fncFdmtU_cu(
		  nullptr      // on-device input image
		, pAuxBuff_fdmt
		, pStreamPars->m_n_p
		, pStreamPars->m_lenChunk / pStreamPars->m_n_p // dimensions of input image 	
		, IDeltaT
		, pStreamPars->m_f_min
		, pStreamPars->m_f_max
		, pStreamPars->m_IMaxDT
		, d_arrfdmt_norm	// OUTPUT image, dim = IDeltaT x IImgcols
		, true
	);
	
	// !6
		
	float* buff = (float*)malloc(2 *sizeof(float) * pStreamPars->m_lenChunk);
	quantOfSuccessfulChunks = 0;

	// 7. variables declatre - remains not readed elements


	int iremains = pStreamPars->m_lenarr;
	float val_coherent_d;
	// !7
	
	// 8.  Array to store cuFFT plans for different array sizes
	cufftResult result;
	//cudaStream_t stream = NULL;
	//cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	
	cufftHandle plan0 = NULL;
	cufftCreate(&plan0);
	
	result = cufftPlan1d(&plan0, pStreamPars->m_lenChunk, CUFFT_C2C, 1);
	if (result != CUFFT_SUCCESS) {
		// Handle error (for simplicity, just print an error message)
		std::cerr << "Error creating cuFFT plan0 \n";
		return -1;
	}

	cufftHandle plan1 = NULL;
	cufftCreate(&plan1);
	int n_p = pStreamPars->m_n_p;
	int lenChunk = pStreamPars->m_lenChunk;
	int n[1] = { n_p };
	//result = cufftPlanMany(&plan1, 1, n, NULL, 1, n_p, NULL, 1, lenChunk / n_p, CUFFT_C2C, lenChunk / n_p);
	result = cufftPlan1d(&plan1, n_p, CUFFT_C2C, lenChunk / n_p);
	if (result != CUFFT_SUCCESS) {
		// Handle error (for simplicity, just print an error message)
		std::cerr << "Error creating cuFFT plan \n";
		return -1;
	}

	
	
	/*for (int i = 0; i < 2; ++i)
	{
		plan_arr[i] = NULL;
	}
	
	createArrayWithPlans(pStreamPars->m_lenChunk, pStreamPars->m_n_p, plan_arr);*/
	
	// !8

	// 9. main loop
	auto start = std::chrono::high_resolution_clock::now();	
	
	
	for (int i = 0; i < NumChunks; ++i)
	{
		int length = (iremains < pStreamPars->m_lenChunk) ? iremains : pStreamPars->m_lenChunk;
		auto start = std::chrono::high_resolution_clock::now();
		
		size_t sz = fread(buff, sizeof(float), 2 * length, pStreamPars->m_stream);

		
		// Convert complex<float> data to cufftComplex and store in pcmparrRawSignalCur
		/*for (int k = 0; k < length; ++k)
		{
			
			pcmparrRawSignalCur[k].x = buff[2 * k];
			pcmparrRawSignalCur[k].y = buff[2 * k + 1];
		}*/

		
		cudaMemcpy(pcmparrRawSignalCur, reinterpret_cast <cufftComplex*> (buff), length * 2 * sizeof(float), cudaMemcpyHostToDevice);
		/*for (int k = 0; k < 10; ++k)
		{

			std::cout << k << "   " << pcmparrRawSignalCur[k].x << "   " << pcmparrRawSignalCur[k].y << endl;
		}*/
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
		iReadTransform_time += duration.count();
		/*std::vector<std::complex<float>> data1(length, 0);
		cudaMemcpy(data1.data(), pcmparrRawSignalCur, length * sizeof(std::complex<float>),
			cudaMemcpyDeviceToHost);*/


		if (fncChunkHybridDedispersion_gpu(pcmparrRawSignalCur, length, pStreamPars->m_n_p
			, pStreamPars->m_D_max, pStreamPars->m_IMaxDT, pStreamPars->m_f_min, pStreamPars->m_f_max
			, pStreamPars->m_SigmaBound, val_coherent_d, pAuxBuff_fdmt
			, pffted_rowsignal 	
			, pcarrTemp
			, pcarrCD_Out
			, pcarrBuff
			, pAuxBuff_flt
			, d_arrfdmt_norm
		    , IDeltaT, plan0, plan1))

		{
			piarrNumSuccessfulChunks[quantOfSuccessfulChunks] = i;
			parrCoherent_d[quantOfSuccessfulChunks] = val_coherent_d;
			++quantOfSuccessfulChunks;
		}
		++(pStreamPars->m_numCurChunk);
		iremains -= pStreamPars->m_lenChunk;
	}
	
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	
	std::cout << "/*****************************************************/ " << std::endl;
	std::cout << "/*****************************************************/ " << std::endl;
	std::cout << "/*****************************************************/ " << std::endl;
	iTotal_time = duration.count();
	std::cout << "Total time:                   " << iTotal_time << " microseconds" << std::endl;
	std::cout << "FDMT time:                    " << iFdmt_time << " microseconds" << std::endl;
	std::cout << "Read and data transform time: " << iReadTransform_time << " microseconds" << std::endl;
	std::cout << "FFT time:                     " << iFFT_time << " microseconds" << std::endl;
	std::cout << "Mean && disp time:            " << iMeanDisp_time << " microseconds" << std::endl;
	std::cout << "Normalization time:           " << iNormalize_time << " microseconds" << std::endl;
	
	std::cout << "/*****************************************************/ " << std::endl;
	std::cout << "/*****************************************************/ " << std::endl;
	std::cout << "/*****************************************************/ " << std::endl;



	cudaFree(pcmparrRawSignalCur);
	cudaFree(d_arrfdmt_norm);
	cudaFree(pAuxBuff_fdmt);	
	cudaFree(pffted_rowsignal); //1	
	cudaFree(pcarrTemp); //2	
	cudaFree(pcarrCD_Out);//3
	cudaFree(pcarrBuff);//3
	cudaFree(pAuxBuff_flt);
	free(buff);
	cufftDestroy(plan0);
	cufftDestroy(plan1);
	/*for (int i = 0; i < 2; ++i)
	{
		cufftDestroy(plan_arr[i]);
		plan_arr[i] = NULL;
	}*/
	return 0;
}
//-------------------------------------------------------------------------------------------------
bool fncChunkHybridDedispersion_gpu(cufftComplex* pcmparrRawSignalCur
	, const unsigned int LEnChunk, const unsigned int N_p
	, const float VAlD_max, const int IMaxDT, const float VAlFmin, const float VAlFmax, float& valSigmaBound_
	, float& coherent_d	,void* pAuxBuff_fdmt
	, cufftComplex* pffted_rowsignal
	, cufftComplex* pcarrTemp 
	, cufftComplex* pcarrCD_Out 
	, cufftComplex* pcarrBuff
	, fdmt_type_* pAuxBuff_flt, fdmt_type_* d_arrfdmt_norm
	, const int IDeltaT, cufftHandle plan0, cufftHandle plan1)
{
	// 1. installation of pointers	for pAuxBuff_the_rest
	//cufftComplex* pffted_rowsignal = ppAuxComp[0];  //1	
	//cufftComplex* pcarrTemp = ppAuxComp[1]; //2	
	//cufftComplex* pcarrCD_Out = ppAuxComp[2]; //3
	//cufftComplex *pcarrBuff =  ppAuxComp[3];

	fdmt_type_* parr_fdmt_inp = pAuxBuff_flt; //4	
	fdmt_type_* parr_fdmt_out = pAuxBuff_flt + LEnChunk; //5

	bool bres = false;
	coherent_d = -1.;
	float valSigmaBound = valSigmaBound_;
	// !1

	/* std::vector<std::complex<float>> data(LEnChunk, 0);	
	cudaMemcpy(data.data(), pffted_rowsignal, LEnChunk * sizeof(std::complex<float>),
		cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	std::array<long unsigned, 1> leshape127{ LEnChunk };
	npy::SaveArrayAsNumpy("ffted.npy", false, leshape127.size(), leshape127.data(), data);*/
	auto start = std::chrono::high_resolution_clock::now();
	
	cufftResult result;
	// 2. create FFT	
	result = cufftExecC2C(plan0, pcmparrRawSignalCur, pffted_rowsignal, CUFFT_FORWARD);
	if (result != CUFFT_SUCCESS) {
		// Handle error (for simplicity, just print an error message)
		std::cerr << "Error executing cuFFT\n";
		exit(EXIT_FAILURE);
	}
	// !2
	
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	iFFT_time += duration.count();

	//std::vector<std::complex<float>> data(LEnChunk, 0);
	//cudaMemcpy(data.data(), pffted_rowsignal, LEnChunk * sizeof(std::complex<float>),
	//	cudaMemcpyDeviceToHost);
	//cudaDeviceSynchronize();
	//cout << "pffted_rowsignal[0] " << data[0].real() << "   " << data[0].imag() << endl;
	//cout << "pffted_rowsignal[1000] " << data[1000].real() << "   " << data[1000].imag() << endl;

	// 3.		
	float valConversionConst = DISPERSION_CONSTANT * (1. / (VAlFmin * VAlFmin) - 1. / (VAlFmax * VAlFmax)) * (VAlFmax - VAlFmin);
	float valN_d = VAlD_max * valConversionConst;
	int n_coherent = int(ceil(valN_d / (N_p * N_p)));
	cout << "n_coherent = " << n_coherent << endl;
	// !3

	fdmt_type_ * d_pmaxSig = nullptr;
	unsigned int* d_pargmaxRow = nullptr;
	unsigned int* d_pargmaxCol = nullptr;

	cudaMallocManaged((void**)&d_pmaxSig, sizeof(fdmt_type_));
	cudaMallocManaged((void**)&d_pargmaxRow, sizeof(unsigned int));
	cudaMallocManaged((void**)&d_pargmaxCol, sizeof(unsigned int));
	// 4. main loop
	//for (int iouter_d = 31; iouter_d < 32; ++iouter_d)
	for (int iouter_d = 0; iouter_d < n_coherent; ++iouter_d)
	{
		cout << "coherent iteration " << iouter_d << endl;

		/*long*/ double valcur_coherent_d = ((/*long*/ double)iouter_d) * ((/*long*/ double)VAlD_max / ((/*long*/ double)n_coherent));
		cout << "cur_coherent_d = " << valcur_coherent_d << endl;
		
		createOutputFDMT_gpu(parr_fdmt_out, pffted_rowsignal, pcarrCD_Out, pcarrTemp
			, LEnChunk, N_p, parr_fdmt_inp, IMaxDT
			, DISPERSION_CONSTANT * valcur_coherent_d, VAlD_max, VAlFmin, VAlFmax, pAuxBuff_fdmt
			, IDeltaT, plan0,plan1, pcarrBuff);
		unsigned int len = (LEnChunk / N_p) * N_p;



		/*std::vector<float> data(LEnChunk, 0);
		cudaMemcpy(data.data(), parr_fdmt_out, LEnChunk * sizeof(float),
			cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		cout << "parr_fdmt_out[0] " << data[0] << "   " << data[1] << endl;
		cout << "parr_fdmt_out[1000] " << data[1000] << "   " << data[1001] << endl;*/
		
		int threadsPerBlock = calcThreadsForMean_and_Disp(len);
		fncSignalDetection_gpu<<<1, threadsPerBlock, threadsPerBlock* (sizeof(fdmt_type_ ) + sizeof(unsigned int)) >> >(
			parr_fdmt_out, d_arrfdmt_norm, LEnChunk / N_p
				, len, d_pmaxSig, d_pargmaxRow, d_pargmaxCol);
		
		fdmt_type_ maxSig = -1;
		unsigned int argmaxRow = 0;
		unsigned int  argmaxCol = 0;
		cudaMemcpy(&maxSig, d_pmaxSig, sizeof(fdmt_type_), cudaMemcpyDeviceToHost);
		cudaMemcpy(&argmaxRow, d_pargmaxRow, sizeof(unsigned int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&argmaxCol, d_pargmaxCol, sizeof(unsigned int), cudaMemcpyDeviceToHost);
		std::cout <<  " maxSig =   " << maxSig <<  endl;
		if (maxSig > valSigmaBound)
		{
			valSigmaBound = maxSig;
			coherent_d = valcur_coherent_d;
			cout << "!!!!!!! achieved score with " << valSigmaBound << "!!!!!!!" << endl;
			bres = true;
			/*if (nullptr != poutImage)
			{
				memcpy(poutImage, parr_fdmt_out, len * sizeof(float));
			}*/
			std::cout << "SNR = " << maxSig << endl;			
			std::cout << "ROW ARGMAX = " << argmaxRow << endl;
			std::cout << "COLUMN ARGMAX = " << argmaxCol << endl;

		}
	}
	
	cudaFree(d_pmaxSig);
	cudaFree(d_pargmaxRow);
	cudaFree(d_pargmaxCol);
	return bres;
	
}


//--------------------------------------------------------------
//INPUT:
// pffted_rowsignal - complex array, ffted 1-dimentional row signal, done from current chunk,  length = LEnChunk
// pcarrCD_Out - memory allocated comlex buffer to save output of coherent dedispersion function, nmed as fncCoherentDedispersion,
//				1- dimentional complex array, length = 	LEnChunk
// pcarrTemp - memory allocated comlex buffer to save output of STFT function, named as fncSTFT. 2-dimentional complex array
//            with dimensions = N_p x (LEnChunk / N_p)
// LEnChunk - length of input ffted signal pffted_rowsignal
// N_p - 
// parr_fdmt_inp - memory allocated float buffer to save input for FDMT function, dimentions = N_p x (LEnChunk / N_p)
// IMaxDT - the maximal delay (in time bins) of the maximal dispersion. Appears in the paper as N_{\Delta}
//            A typical input is maxDT = N_f
// VAlLong_coherent_d - is DispersionConstant* d, where d - is the dispersion measure.units: pc * cm ^ -3
// VAlD_max - maximal dispersion to scan, in units of pc cm^-3
// VAlFmin - the minimum freq, given in Mhz
// VAlFmax - the maximum freq,
//
// OUTPUT:
// parr_fdmt_out - float 2-dimensional array,with dimensions =  IMaxDT x (LEnChunk / N_p)
int createOutputFDMT_gpu(fdmt_type_* parr_fdmt_out, cufftComplex* pffted_rowsignal, cufftComplex* pcarrCD_Out
	, cufftComplex* pcarrTemp, const unsigned int LEnChunk, const unsigned int N_p, fdmt_type_* d_parr_fdmt_inp
	, const unsigned int IMaxDT, const /*long*/ double VAl_practicalD, const float VAlD_max, const float VAlFmin
	, const float VAlFmax, void* pAuxBuff_fdmt, const int IDeltaT, cufftHandle plan0, cufftHandle plan1
	,  cufftComplex* pAuxBuff)
{	 
	fncCoherentDedispersion_gpu(pcarrCD_Out, pffted_rowsignal, LEnChunk, VAl_practicalD, VAlFmin, VAlFmax
	, plan0, pcarrTemp);

	/*std::vector<std::complex<float>> data(LEnChunk, 0);
	cudaMemcpy(data.data(), pcarrCD_Out, LEnChunk * sizeof(std::complex<float>),
		cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cout << "pcarrCD_Out[0] " << data[0].real() << "   " << data[0].imag() << endl;
	cout << "pcarrCD_Out[1000] " << data[1000].real() << "   " << data[1000].imag() << endl;*/


	fncSTFT_gpu(pcarrTemp, pcarrCD_Out, LEnChunk, N_p,  plan1, pAuxBuff);

	calc_fdmt_inp(d_parr_fdmt_inp, pcarrTemp, N_p, LEnChunk / N_p
		, pAuxBuff_fdmt);

	auto start = std::chrono::high_resolution_clock::now();
	

	fncFdmtU_cu(d_parr_fdmt_inp      // on-device input image
		, pAuxBuff_fdmt
		, N_p
		, LEnChunk / N_p // dimensions of input image 	
		, IDeltaT
		, VAlFmin
		, VAlFmax
		, IMaxDT
		, parr_fdmt_out	// OUTPUT image, dim = IDeltaT x IImgcols
		, false
	);
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	iFdmt_time += duration.count();
	return 0;
}
//--------------------------------------------------------------------
__global__
void kernel_create_arr_fdmt_inp(fdmt_type_* d_parr_fdmt_inp, cufftComplex* d_mtrxPower, unsigned int len
	, const float val_mean, const float val_stdDev)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= len)
	{
		return;
	}
	d_parr_fdmt_inp[i] = (fdmt_type_)((d_mtrxPower[i].x - val_mean) / (0.25 * val_stdDev));

}
//--------------------------------------------------------------------

void calc_fdmt_inp(fdmt_type_* d_parr_fdmt_inp, cufftComplex* pcarrTemp, unsigned int nRows, unsigned int nCols
	,void* pAuxBuff)
{
	
	float* pval_mean = nullptr;
	float* pval_stdDev = nullptr;
	cudaMalloc(reinterpret_cast<void**>(&pval_mean), sizeof(float ));
	cudaMalloc(reinterpret_cast<void**>(&pval_stdDev), sizeof(float));

	auto start = std::chrono::high_resolution_clock::now();
	
	// Calculate mean and variance
	int blocksPerGrid = nRows;	
	int treadsPerBlock = calcThreadsForMean_and_Disp(nCols);
	size_t sz = (2 * sizeof(float) + sizeof(int))* treadsPerBlock;
	float* d_arrSumMean = (float*)((char*)pAuxBuff);// +nRows * nCols * sizeof(fdmt_type_));
	float* d_arrSumMeanSquared = d_arrSumMean + nRows;
	calcPowerMtrx_and_RowMeans_and_Disps << < blocksPerGrid, treadsPerBlock, sz >> >
		(d_parr_fdmt_inp,  pcarrTemp, nRows, nCols, d_arrSumMean, d_arrSumMeanSquared);
	cudaDeviceSynchronize();
	
	
	blocksPerGrid = 1;
	treadsPerBlock = calcThreadsForMean_and_Disp(nRows);
	sz = treadsPerBlock * (2 *sizeof(float) + sizeof(int));
	kernel_OneSM_Mean_and_Disp << <blocksPerGrid, treadsPerBlock, sz >> > (d_arrSumMean, d_arrSumMeanSquared, nRows
		, pval_mean, pval_stdDev);
	cudaDeviceSynchronize();

	float mean = -1., disp = -1.;
	cudaMemcpy(&mean, pval_mean, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&disp, pval_stdDev, sizeof(float), cudaMemcpyDeviceToHost);

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	iMeanDisp_time += duration.count();

	blocksPerGrid = (nRows * nCols + treadsPerBlock - 1) / treadsPerBlock;
	
	kernel_normalize_array<<<blocksPerGrid, treadsPerBlock >>>(d_parr_fdmt_inp, nRows * nCols
		, pval_mean, pval_stdDev);
	cudaDeviceSynchronize();

	auto end1 = std::chrono::high_resolution_clock::now();
	auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - end);
	iNormalize_time += duration1.count();

}

//-----------------------------------------------------------------
__global__ void calcPowerMtrx_and_RowMeans_and_Disps
(fdmt_type_* d_parr_fdmt_inp_, cufftComplex* pcarrTemp_,const int NRows, const int NCols
	, float* d_arrSumMean, float* d_arrSumMeanSquared)
{
	extern __shared__ float sbuff[];
	float* sdata = sbuff;
	int* sNums = (int*)((char*)sbuff + 2 * blockDim.x * sizeof(float));
	
	cufftComplex* pcarrTemp = pcarrTemp_ + NCols * blockIdx.x + threadIdx.x;
	fdmt_type_* d_parr_fdmt_inp = d_parr_fdmt_inp_ + NCols * blockIdx.x + threadIdx.x;
	unsigned int tid = threadIdx.x;
	unsigned int i = tid;// blockIdx.x* blockDim.x + threadIdx.x;
	if (tid >= NCols)
	{
		sNums[tid] = 0;
		sdata[tid] = 0.0f;
		sdata[blockDim.x + tid] = 0.0f;
	}
	else
	{
		float localSum = 0.0f;
		float localSquaredSum = 0.0f;
		int numLocal = 0;
		// Calculate partial sums within each block   
		while (i < NCols)
		{
			float temp = (*pcarrTemp).x * (*pcarrTemp).x + (*pcarrTemp).y * (*pcarrTemp).y;
			(*d_parr_fdmt_inp) = (fdmt_type_)temp;
			localSum += temp;
			localSquaredSum += temp * temp;
			i += blockDim.x;
			pcarrTemp += blockDim.x;
			d_parr_fdmt_inp += blockDim.x;
			++numLocal;
		}

		// Store partial sums in shared memory	
		sNums[tid] = numLocal;
		sdata[tid] = localSum / numLocal;
		sdata[blockDim.x + tid] = localSquaredSum / numLocal;
	}
	__syncthreads();

	// Parallel reduction within the block to sum partial sums
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
		{			
			sdata[tid] = (sdata[tid] * sNums[tid] + sdata[tid + s] * sNums[tid + s]) / (sNums[tid] + sNums[tid + s]);
			sdata[blockDim.x + tid] = (sdata[blockDim.x + tid] * sNums[tid] + sdata[blockDim.x + tid + s] * sNums[tid + s])
				/ (sNums[tid] + sNums[tid + s]);
			sNums[tid] = sNums[tid] + sNums[tid + s];
		}
		__syncthreads();
	}

	// Only thread 0 within each block computes the block's sum
	if (tid == 0)
	{
		d_arrSumMean[blockIdx.x] = sdata[0];
		d_arrSumMeanSquared[blockIdx.x] = sdata[blockDim.x] ;
	}
	__syncthreads();

}
//-----------------------------------------------------------------
__global__ void kernel_OneSM_Mean_and_Disp(float* d_arrMeans, float* d_arrDisps, int len
	, float* pvalMean, float* pvalDisp)
{
	extern __shared__ float sbuff[];
	float* sdata = sbuff;
	int* sNums = (int*)((char*)sbuff + 2 * blockDim.x * sizeof(float));

	unsigned int tid = threadIdx.x;
	unsigned int i = tid;
	if (tid >= len)
	{
		sNums[tid] = 0;
		sdata[tid] = 0;
		sdata[blockDim.x + tid] = 0;
	}
	else
	{
		float localSum0 = 0.0f;
		float localSum1 = 0.0f;
		int numLocal = 0;
		// Calculate partial sums within each block   
		while (i < len)
		{
			localSum0 += d_arrMeans[i];
			localSum1 += d_arrDisps[i];
			i += blockDim.x;
			++numLocal;
		}

		// Store partial sums in shared memory	
		sNums[tid] = numLocal;
		sdata[tid] = localSum0 / numLocal;
		sdata[blockDim.x + tid] = localSum1 / numLocal;
	}
	__syncthreads();

	// Parallel reduction within the block to sum partial sums
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s) 
		{
			sdata[tid] = (sdata[tid] * sNums[tid] + sdata[tid + s] * sNums[tid + s]) / (sNums[tid] + sNums[tid + s]);
			sdata[blockDim.x + tid] = (sdata[blockDim.x + tid] * sNums[tid] + sdata[blockDim.x + tid + s] * sNums[tid + s])
				/ (sNums[tid] + sNums[tid + s]);
			sNums[tid] = sNums[tid] + sNums[tid + s];
		}
		__syncthreads();
	}

	// Only thread 0 within each block computes the block's sum
	if (tid == 0)
	{
		*pvalMean =  sdata[0];
		*pvalDisp = sqrt(sdata[blockDim.x] -sdata[0] * sdata[0]);

	}
	__syncthreads();
}
//---------------------------------------------------------------------------------------------------------
__global__
void fncSignalDetection_gpu(fdmt_type_ * parr_fdmt_out, fdmt_type_ * parrImNormalize, const unsigned int qCols
	, const unsigned int len, fdmt_type_ *pmaxElement, unsigned int* argmaxRow, unsigned int* argmaxCol)
{
	extern __shared__ char sbuff1[];
	fdmt_type_* sdata = (fdmt_type_ * )sbuff1;
	int* sNums = (int*)((char*)sbuff1 + blockDim.x * sizeof(fdmt_type_));

	unsigned int tid = threadIdx.x;
	unsigned int i = tid;
	if (tid >= len)
	{
		sNums[tid] = -1;
		sdata[tid] = -FLT_MAX;
	}
	else
	{
		fdmt_type_ localMax = -FLT_MAX;
		int localArgMax = 0;

		// Calculate partial sums within each block   
		while (i < len)
		{
			fdmt_type_ t = parr_fdmt_out[i] / sqrt(parrImNormalize[i] * 16 + 0.000001);
			parr_fdmt_out[i] = t;
			if (t > localMax)
			{
				localMax = t;
				localArgMax = i;
			}
			i += blockDim.x;
		}
		// Store partial sums in shared memory
		//numLocal = len / blockDim.x;
		sNums[tid] = localArgMax;
		sdata[tid] = localMax;
	}
	__syncthreads();

	// Parallel reduction within the block to sum partial sums
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			if (sdata[tid + s] > sdata[tid])
			{
				sdata[tid] = sdata[tid + s];
				sNums[tid] = sNums[tid + s];
			}
			
		}
		__syncthreads();
	}
	// Only thread 0 within each block computes the block's sum
	if (tid == 0)
	{
		*pmaxElement = sdata[0];
		*argmaxRow = sNums[0] / qCols;
		*argmaxCol = sNums[0] % qCols;

	}
	__syncthreads();
}
//---------------------------------------------------------------------
__global__ void kernel_normalize_array  (fdmt_type_* pAuxBuff,const unsigned int len
	, float  *pmean, float *pdev)
{
	unsigned int i =  blockIdx.x* blockDim.x + threadIdx.x;
	if (i >= len)
	{
		return;
	}
	pAuxBuff[i] = (pAuxBuff[i] - (*pmean)) / ((*pdev) * 0.25);

}
//-------------------------------------------------------------------

void fncSTFT_gpu(cufftComplex* pcarrOut, cufftComplex* pRawSignalCur, const unsigned int LEnChunk, int block_size
	, cufftHandle plan_short, cufftComplex* pAuxBuff)
{
	int qRows = LEnChunk / block_size;	
	
	auto start = std::chrono::high_resolution_clock::now();
	
	cufftResult result = cufftExecC2C(plan_short, pRawSignalCur, pAuxBuff, CUFFT_FORWARD);
	cudaDeviceSynchronize();	
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	iFFT_time += duration.count();	
	

	dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
	dim3 blocksPerGrid((block_size + TILE_DIM - 1) / TILE_DIM, (qRows + TILE_DIM - 1) / TILE_DIM);
	size_t sz = TILE_DIM * (TILE_DIM + 1) * sizeof(cufftComplex);

	transpose << <blocksPerGrid, threadsPerBlock,sz >> > (pAuxBuff, pcarrOut, block_size, qRows);
	
}
//-----------------------------------------------------------------
//INPUT:
// d_mtrxSig - input matrix type of cufftComplex
// len - number elements of matrix
// OUTPUT:
// pcarrTemp[j].x = pcarrTemp[j].x * pcarrTemp[j].x + pcarrTemp[j].y * pcarrTemp[j].y;
// psum = sum (pcarrTemp[j].x)
__global__
void kernel_Aux(float* psum, float* psumSq, cufftComplex* d_mtrxSig
	, unsigned int len)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= len)
	{
		return;
	}
	d_mtrxSig[i].x = d_mtrxSig[i].x * d_mtrxSig[i].x + d_mtrxSig[i].y * d_mtrxSig[i].y;
	atomicAdd(psum, d_mtrxSig[i].x);
	atomicAdd(psumSq, d_mtrxSig[i].x * d_mtrxSig[i].x);
}


//-------------------------------------------------------------------
void fncCoherentDedispersion_gpu(cufftComplex* pcarrCD_Out, cufftComplex* pcarrffted_rowsignal
	, const unsigned int LEnChunk, const /*long*/ double VAl_practicalD, const float VAlFmin, const float VAlFmax
    , cufftHandle  plan, cufftComplex* pAuxBuff)
{
	double step = (( double)VAlFmax - ( double)VAlFmin) / (( double)LEnChunk);
	
	int treadsPerBlock = 1024;
	int blocksPerGrid = (LEnChunk + treadsPerBlock -1)/ treadsPerBlock;

	
	/*double* arr_t4 = NULL;
	cudaMalloc(&arr_t4, LEnChunk * sizeof(double));
	kernel_Temp0 << < blocksPerGrid, treadsPerBlock >> > (arr_t4, LEnChunk, step, VAl_practicalD, VAlFmin
		, VAlFmax);
	cudaDeviceSynchronize();
	std::vector<double> data(LEnChunk, 0);
	cudaMemcpy(data.data(), arr_t4, LEnChunk * sizeof(double),
		cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cout << "arr_t4 [0] now " << data[0] << "   " << data[1] << endl;
	cout << "arr_t4 [1000] now " << data[1000] << "   " << data[1001] << endl;

	kernel_Temp1 << < blocksPerGrid, treadsPerBlock >> > (pAuxBuff, pcarrffted_rowsignal, arr_t4
		, LEnChunk, step, VAl_practicalD, VAlFmin
		, VAlFmax);
	cudaDeviceSynchronize();
	cudaFree(arr_t4);*/
	kernel_calcAuxArray << < blocksPerGrid, treadsPerBlock >> >
		(pAuxBuff, pcarrffted_rowsignal, LEnChunk, step, VAl_practicalD, VAlFmin
			,  VAlFmax);
	cudaDeviceSynchronize();
	/*std::vector<std::complex<float>> data2(LEnChunk, 0);
	cudaMemcpy(data2.data(), pAuxBuff, LEnChunk * sizeof(std::complex<float>),
		cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cout << "pAuxBuff[0] now " << data2[0].real() << "   " << data2[0].imag() << endl;
	cout << "pAuxBuff[1000] now " << data2[1000].real() << "   " << data2[1000].imag() << endl;*/
	auto start = std::chrono::high_resolution_clock::now();
	
	cufftResult result = cufftExecC2C(plan, pAuxBuff, pcarrCD_Out, CUFFT_INVERSE);

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	iFFT_time += duration.count();

	/*std::vector<std::complex<float>> data1(LEnChunk, 0);
	cudaMemcpy(data1.data(), pcarrCD_Out, LEnChunk * sizeof(std::complex<float>),
		cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cout << "!!pcarrCD_Out0[0] " << data1[0].real() << "   " << data1[0].imag() << endl;
	cout << "!!pcarrCD_Out0[1000] " << data1[1000].real() << "   " << data1[1000].imag() << endl;*/

	
	scaling_kernel << <1, 1024, 0 >> > (pcarrCD_Out, LEnChunk, 1.f / ((float)LEnChunk));
	cudaDeviceSynchronize();
}

//-------------------------------------------------------------
 // calculation:
 // H = np.e**(-(2*np.pi*complex(0,1) * practicalD /(f_min + f) + 2*np.pi*complex(0,1) * practicalD*f /(f_max**2)))
 // np.fft.fft(raw_signal) * H 
__global__ void kernel_calcAuxArray(cufftComplex* pAuxBuff, cufftComplex* pcarrffted_rowsignal
	, const unsigned int LEnChunk, const /*long*/ double step, const /*long*/ double VAl_practicalD, const double fmin
	, const double fmax)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= LEnChunk)
	{
		return;
	}
	
	double t = VAl_practicalD * (1. / ((double)fmin + step * (double)i) + 1. / ((double)fmax) * (step * (double)i / (double)fmax));

	double val_prD_int = 0, val_prD_frac = 0;
	double t4 = -modf(t, &val_prD_int) *2.0;

	double val_x = 0.;
	double val_y = 0.;
	sincospi(t4, &val_y, &val_x);
	pAuxBuff[i].x = (float)(val_x * pcarrffted_rowsignal[i].x - val_y * pcarrffted_rowsignal[i].y); // Real part
	pAuxBuff[i].y = (float)(val_x * pcarrffted_rowsignal[i].y + val_y * pcarrffted_rowsignal[i].x); // Imaginary part
	 
}
//--------------------------------------------------
__global__ void kernel_Temp0( double *arr_t4
	, const unsigned int LEnChunk, const /*long*/ double step, const /*long*/ double VAl_practicalD, const double fmin
	, const double fmax)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= LEnChunk)
	{
		return;
	}

	/*double delfi = step * (double)i ;
	double val_fi = fmin  + delfi;
	double valPract = VAl_practicalD * 1.0E-3;
	double t2 = valPract *(1. / val_fi
		+ 1. / (fmax) * (delfi / fmax)) * 2. * M_PI;


	double t3 = remainder(t2, 2. * M_PI);
	t3 *= 1.0E3;
	double val_x = 0.;
	double val_y = 0.;*/

	double val_prD_int = 0, val_prD_frac = 0;
	val_prD_frac = my_modf((double)VAl_practicalD, &val_prD_int);
	
	arr_t4[i] = val_prD_frac;
	//double val_prX_int = 0, val_prX_frac = 0;
	//val_prX_frac = modf(1. / (fmin + step * (double)i) + 1. / (fmax) * (step * (double)i / fmax), &val_prX_int);
	//double t0 = fmod((1000.0 * val_prD_frac) * (1000.0 * val_prX_frac), 1.0E6) / 1.0E6;
	//double t1 = fmod(1000. * val_prX_frac * val_prD_int, 1.0E3) / 1.0E3;
	//double t2 = 0.;// fmod(1000. * val_prD_frac * val_prX_int, 1.0E3) / 1.0E3;
	//double t4 = -(t0 + t1 + t2) * 2.;
	//arr_t4[i] = t4;
}
//------------------------------------------------------------------------
__global__ void kernel_Temp1(cufftComplex* pAuxBuff, cufftComplex* pcarrffted_rowsignal, double *arr_t4
	, const unsigned int LEnChunk, const /*long*/ double step, const /*long*/ double VAl_practicalD, const double fmin
	, const double fmax)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= LEnChunk)
	{
		return;
	}

	
	double val_x = 0.;
	double val_y = 0.;
	sincospi(arr_t4[i], &val_y, &val_x);
	double d0 = (float)(val_x * pcarrffted_rowsignal[i].x - val_y * pcarrffted_rowsignal[i].y); // Real part
	double d1 = (float)(val_x * pcarrffted_rowsignal[i].y + val_y * pcarrffted_rowsignal[i].x); // Imaginary part
	pAuxBuff[i].x = (float)d0; // Real part
	pAuxBuff[i].y = (float)d1; // Imaginary part

}
//---------------------------------------------------------------
__global__
void scaling_kernel(cufftComplex* data, int element_count, float scale)
{
	const int tid = threadIdx.x ;
	const int stride = blockDim.x ;
	for (int i = tid; i < element_count; i += stride)
	{
		data[i].x *= scale;
		data[i].y *= scale;
	}
}
//-----------------------------------------------------------
int createArrayWithPlans(unsigned int lenChunk, unsigned int n_p, cufftHandle* plan_arr)
{
	
	if (plan_arr[0])
	{
		cufftDestroy(plan_arr[0]);
	}
	cufftResult result;
	cufftCreate(&plan_arr[0]);
	result = cufftPlan1d(plan_arr, lenChunk, CUFFT_C2C, 1);
	if (result != CUFFT_SUCCESS) {
		// Handle error (for simplicity, just print an error message)
		std::cerr << "Error creating cuFFT plan \n";
		return -1;
	}

	int n[1] = { n_p };
	cufftCreate(&plan_arr[1]);
	result = cufftPlanMany(&plan_arr[1], 1, n, NULL, 1, n_p, NULL, 1, lenChunk / n_p, CUFFT_C2C, lenChunk/n_p);
	if (result != CUFFT_SUCCESS) {
		std::cerr << "Error creating cuFFT cufftPlanMany\n";		
		return -1;
	}
	return 0;
}
//---------------------------------------------------
__global__ void transpose(cufftComplex* input, cufftComplex* output, int width, int height)
{
	__shared__ cufftComplex tile[TILE_DIM][TILE_DIM + 1]; // Shared memory tile

	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;

	// Transpose data from global to shared memory
	if (x < width && y < height) {
		tile[threadIdx.y][threadIdx.x] = input[y * width + x];
	}
	__syncthreads();

	// Calculate new indices for writing to output
	x = blockIdx.y * TILE_DIM + threadIdx.x;
	y = blockIdx.x * TILE_DIM + threadIdx.y;

	// Transpose data from shared to global memory
	if (x < height && y < width) {
		output[y * height + x] = tile[threadIdx.x][threadIdx.y];
	}
}
//---------------------------------------------------
int malloc_for_4_complex_arrays(cufftComplex** ppffted_rowsignal, cufftComplex** ppcarrTemp
	, cufftComplex** ppcarrCD_Out, cufftComplex** ppcarrBuff,const unsigned int LEnChunk)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc((void**)ppffted_rowsignal, LEnChunk * sizeof(cufftComplex));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "pffted_rowsignal. cudaMalloc failed!");
		return 1;
	}
	cudaStatus = cudaMalloc((void**)ppcarrTemp, LEnChunk * sizeof(cufftComplex));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "pcarrTemp. cudaMalloc failed!");
		return 1;
	}
	cudaStatus = cudaMalloc((void**)ppcarrCD_Out, LEnChunk * sizeof(cufftComplex));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "pcarrCD_Out. cudaMalloc failed!");
		return 1;
	}
	cudaStatus = cudaMalloc((void**)ppcarrBuff, LEnChunk * sizeof(cufftComplex));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "pcarrBuff. cudaMalloc failed!");
		return 1;
	}
	return 0;
}

//-------------------------------------------------------------
bool createOutImageForFixedNumberChunk_gpu(fdmt_type_ ** pparr_fdmt_out, int* pargmaxRow, int* pargmaxCol, fdmt_type_ *pvalSNR
	, fdmt_type_ ** pparrOutSubImage, int* piQuantRowsPartImage, CStreamParams* pStreamPars, const int numChunk
	, const float VAlCoherent_d)
{

	cudaError_t cudaStatus;
	const int iremains = pStreamPars->m_lenarr - numChunk * pStreamPars->m_lenChunk;
	if (iremains <= 0)
	{
		fprintf(stderr, "Define chunk's number correctly.");
		return false;
	}
	const unsigned int LEnChunk = (iremains < pStreamPars->m_lenChunk) ? iremains : pStreamPars->m_lenChunk;
	int icols = LEnChunk / pStreamPars->m_n_p;



	// 1. memory allocation for  chunk
	
	cufftComplex* pcmparrRawSignalCur = NULL;
	cudaStatus = cudaMallocManaged((void**)&pcmparrRawSignalCur, LEnChunk * sizeof(cufftComplex));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "fncHybridDedispersionStream_gpu. cudaMallocManaged failed!");
		return false;
	}

	// 1!

	// 2. memory allocation for fdmt_ones on GPU
	fdmt_type_* d_arrfdmt_norm = 0;
	cudaStatus = cudaMallocManaged((void**)&d_arrfdmt_norm, LEnChunk * sizeof(fdmt_type_));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "fncHybridDedispersionStream_gpu. cudaMallocManaged failed!");
		return false;
	}

	// 2!

	// 3.memory allocation for auxillary buffer for fdmt
	const int  IDeltaT = calc_IDeltaT(pStreamPars->m_n_p, pStreamPars->m_f_min, pStreamPars->m_f_max, pStreamPars->m_IMaxDT);
	size_t szBuff_fdmt = calcSizeAuxBuff_fdmt(pStreamPars->m_n_p, LEnChunk / pStreamPars->m_n_p
		, pStreamPars->m_f_min, pStreamPars->m_f_max, pStreamPars->m_IMaxDT);
	void* pAuxBuff_fdmt = 0;
	cudaStatus = cudaMalloc(&pAuxBuff_fdmt, szBuff_fdmt);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "fncHybridDedispersionStream_gpu. cudaMallocManaged for pAuxBuff_fdmt failed!");
		return 1;
	}
	// 3!

	// 4. memory allocation for the 4 auxillary cufftComplex  arrays on GPU

	cufftComplex* pffted_rowsignal = NULL; //1	
	cufftComplex* pcarrTemp = NULL; //2	
	cufftComplex* pcarrCD_Out = NULL;//3
	cufftComplex* pcarrBuff = NULL;//3
	if (1 == malloc_for_4_complex_arrays(&pffted_rowsignal, &pcarrTemp, &pcarrCD_Out, &pcarrBuff, LEnChunk))
	{
		return 1;
	}

	// !4

	

	// 6. calculation fdmt ones
	fncFdmtU_cu(
		nullptr      // on-device input image
		, pAuxBuff_fdmt
		, pStreamPars->m_n_p
		, LEnChunk / pStreamPars->m_n_p // dimensions of input image 	
		, IDeltaT
		, pStreamPars->m_f_min
		, pStreamPars->m_f_max
		, pStreamPars->m_IMaxDT
		, d_arrfdmt_norm	// OUTPUT image, dim = IDeltaT x IImgcols
		, true
	);

	// !6

	
	


	//// 8.  Array to store cuFFT plans for different array sizes
	//cufftHandle plan_arr[2];
	//for (int i = 0; i < 2; ++i)
	//{
	//	plan_arr[i] = NULL;
	//}

	//createArrayWithPlans(LEnChunk, pStreamPars->m_n_p, plan_arr);

	// !8
	// 8.  Array to store cuFFT plans for different array sizes
	cufftResult result;
	cufftHandle plan0 = NULL;
	cufftCreate(&plan0);
	result = cufftPlan1d(&plan0, LEnChunk, CUFFT_C2C, 1);
	if (result != CUFFT_SUCCESS) {
		// Handle error (for simplicity, just print an error message)
		std::cerr << "Error creating cuFFT plan \n";
		return -1;
	}

	cufftHandle plan1 = NULL;
	cufftCreate(&plan1);
	int n_p = pStreamPars->m_n_p;
	int lenChunk = pStreamPars->m_lenChunk;
	int n[1] = { n_p };
	result = cufftPlanMany(&plan1, 1, n, NULL, 1, n_p, NULL, 1, lenChunk / n_p, CUFFT_C2C, lenChunk / n_p);
	if (result != CUFFT_SUCCESS) {
		// Handle error (for simplicity, just print an error message)
		std::cerr << "Error creating cuFFT plan \n";
		return -1;
	}


	// 9. read input data from file
	    fseek(pStreamPars->m_stream, numChunk* pStreamPars->m_lenChunk *2 * sizeof(float), SEEK_CUR);
		float* buff = (float*)malloc(2 * sizeof(float) * LEnChunk);
		//size_t sz = fread(pcmparrRawSignalCur, sizeof(cufftComplex), length, pStreamPars->m_stream);
		size_t sz = fread(buff, sizeof(float), 2 * LEnChunk, pStreamPars->m_stream);
		// Convert complex<float> data to cufftComplex and store in pcmparrRawSignalCur
		for (int k = 0; k < LEnChunk; ++k)
		{

			pcmparrRawSignalCur[k].x = buff[2 * k];
			pcmparrRawSignalCur[k].y = buff[2 * k + 1];
		}

		/*std::vector<std::complex<float>> data1(length, 0);
		cudaMemcpy(data1.data(), pcmparrRawSignalCur, length * sizeof(std::complex<float>),
			cudaMemcpyDeviceToHost);*/
		int IMaxDT = pStreamPars->m_IMaxDT;
		int N_p = pStreamPars->m_n_p;
		const float VAlFmin = pStreamPars->m_f_min;
		const float VAlFmax = pStreamPars->m_f_max;
		const float VAlD_max = pStreamPars->m_D_max;
		

		// 5. memory allocation for the 2 auxillary float  arrays on GPU

		fdmt_type_ * parr_fdmt_inp = NULL;
		cudaStatus = cudaMalloc((void**)&parr_fdmt_inp, LEnChunk * sizeof(fdmt_type_));
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "pffted_rowsignal. cudaMalloc failed!");
			return 1;
		}

		fdmt_type_ * parr_fdmt_out = NULL;
		cudaStatus = cudaMalloc((void**)&parr_fdmt_out, LEnChunk * sizeof(fdmt_type_));
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "pffted_rowsignal. cudaMalloc failed!");
			return 1;
		}
		// 5!
		

		bool bres = false;
		

		
		// 2. create FFT	
		result = cufftExecC2C(plan0, pcmparrRawSignalCur, pffted_rowsignal, CUFFT_FORWARD);
		if (result != CUFFT_SUCCESS) {
			// Handle error (for simplicity, just print an error message)
			std::cerr << "Error executing cuFFT\n";
			exit(EXIT_FAILURE);
		}
		// !2

		// 3.		
		/*float valConversionConst = DISPERSION_CONSTANT * (1. / (VAlFmin * VAlFmin) - 1. / (VAlFmax * VAlFmax)) * (VAlFmax - VAlFmin);
		float valN_d = VAlD_max * valConversionConst;
		int n_coherent = int(ceil(valN_d / (N_p * N_p)));
		cout << "n_coherent = " << n_coherent << endl;*/
		// !3

		fdmt_type_* d_pmaxSig = nullptr;
		unsigned int* d_pargmaxRow = nullptr;
		unsigned int* d_pargmaxCol = nullptr;

		cudaMallocManaged((void**)&d_pmaxSig, sizeof(fdmt_type_));
		cudaMallocManaged((void**)&d_pargmaxRow, sizeof(unsigned int));
		cudaMallocManaged((void**)&d_pargmaxCol, sizeof(unsigned int));
		// 4. main loop
		
			

			
			

			createOutputFDMT_gpu(parr_fdmt_out, pffted_rowsignal, pcarrCD_Out, pcarrTemp
				, LEnChunk, N_p, parr_fdmt_inp, IMaxDT
				, DISPERSION_CONSTANT * VAlCoherent_d, VAlD_max, VAlFmin, VAlFmax, pAuxBuff_fdmt
				, IDeltaT, plan0, plan1, pcarrBuff);
			unsigned int len = (LEnChunk / N_p) * N_p;

			int threadsPerBlock = calcThreadsForMean_and_Disp(len);
			fncSignalDetection_gpu << <1, threadsPerBlock, threadsPerBlock* (sizeof(fdmt_type_) + sizeof(unsigned int)) >> > (parr_fdmt_out, d_arrfdmt_norm, LEnChunk / N_p
				, len, d_pmaxSig, d_pargmaxRow, d_pargmaxCol);

			
			cudaMemcpy(pvalSNR, d_pmaxSig, sizeof(fdmt_type_), cudaMemcpyDeviceToHost);
			cudaMemcpy(pargmaxRow, d_pargmaxRow, sizeof(unsigned int), cudaMemcpyDeviceToHost);
			cudaMemcpy(pargmaxCol, d_pargmaxCol, sizeof(unsigned int), cudaMemcpyDeviceToHost);
			std::cout << "SNR = " << *pvalSNR << endl;
			std::cout << "ROW ARGMAX = " << *pargmaxRow << endl;
			std::cout << "COLUMN ARGMAX = " << *pargmaxCol << endl;
			

		cudaFree(d_pmaxSig);
		cudaFree(d_pargmaxRow);
		cudaFree(d_pargmaxCol);

		// creation parr_fdmt_out
		*pparr_fdmt_out = (fdmt_type_ *)realloc(*pparr_fdmt_out,LEnChunk * sizeof(fdmt_type_));
		cudaMemcpy(*pparr_fdmt_out, parr_fdmt_out, LEnChunk * sizeof(fdmt_type_), cudaMemcpyDeviceToHost);
		cutQuadraticSubImage(pparrOutSubImage, piQuantRowsPartImage, *pparr_fdmt_out, N_p, (LEnChunk / N_p), *pargmaxRow, *pargmaxCol);


	cudaFree(pcmparrRawSignalCur);
	cudaFree(d_arrfdmt_norm);
	cudaFree(pAuxBuff_fdmt);
	cudaFree(pffted_rowsignal); //1	
	cudaFree(pcarrTemp); //2	
	cudaFree(pcarrCD_Out);//3
	cudaFree(pcarrBuff);//3
	cudaFree(parr_fdmt_inp);
	cudaFree(parr_fdmt_out);
	free(buff);
	/*for (int i = 0; i < 2; ++i)
	{
		cufftDestroy(plan_arr[i]);
		plan_arr[i] = NULL;
	}*/

	cufftDestroy(plan0);
	cufftDestroy(plan1);
	return 0;
	

}
//-----------------------------------------------------------------------------------------
void cutQuadraticSubImage(fdmt_type_ ** pparrOutImage, int* piQuantRowsOutImage, fdmt_type_* InpImage
	, const int QInpImageRows, const int QInpImageCols	, const int NUmCentralElemRow, const int NUmCentralElemCol)
{
	*piQuantRowsOutImage = (QInpImageRows < QInpImageCols) ? QInpImageRows : QInpImageCols;
	(*pparrOutImage) = (fdmt_type_ *)realloc((*pparrOutImage), (*piQuantRowsOutImage) * (*piQuantRowsOutImage) * sizeof(fdmt_type_ ));
	fdmt_type_ * p = (*pparrOutImage);
	if (QInpImageRows < QInpImageCols)
	{
		int numPart = NUmCentralElemCol / QInpImageRows;
		int numColStart = numPart * QInpImageRows;
		for (int i = 0; i < QInpImageRows; ++i)
		{
			memcpy(&p[i * QInpImageRows], &InpImage[i * QInpImageCols + numColStart], QInpImageRows * sizeof(fdmt_type_));
		}
		return;
	}
	int numPart = NUmCentralElemRow / QInpImageCols;
	int numStart = numPart * QInpImageCols;
	memcpy(p, &InpImage[numStart], QInpImageCols * QInpImageCols * sizeof(fdmt_type_));
}
//------------------------------------------
__device__ double my_modf(double val, double* intpart)
{
	int it = (int)val;
	*intpart = (double)(it);
		
	return val - (*intpart);
}

