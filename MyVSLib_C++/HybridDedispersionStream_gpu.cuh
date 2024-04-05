#include "StreamParams.h"
#include <thrust/complex.h>
#include <cufft.h>
#include "Constants.h"
//#include <complex>

using namespace std;

class CStreamParams;
int fncHybridDedispersionStream_gpu( int* piarrNumSuccessfulChunks, float* parrCoherent_d, int& quantOfSuccessfulChunks
	, CStreamParams* pStreamPars);

bool fncChunkHybridDedispersion_gpu(cufftComplex* pcmparrRawSignalCur
	, const unsigned int LEnChunk, const unsigned int N_p
	, const float VAlD_max, const int IMaxDT, const float VAlFmin, const float VAlFmax, float& valSigmaBound_
	, float& coherent_d, void* pAuxBuff_fdmt
	, cufftComplex* pffted_rowsignal
	, cufftComplex* pcarrTemp
	, cufftComplex* pcarrCD_Out
	, cufftComplex* pcarrBuff
	, fdmt_type_* pAuxBuff_flt, fdmt_type_* d_arrfdmt_norm
	, const int IDeltaT, cufftHandle plan0, cufftHandle plan1);

int createArrayWithPlans(unsigned int lenChunk, unsigned int n_p, cufftHandle* plan_arr);

int createOutputFDMT_gpu(fdmt_type_* parr_fdmt_out, cufftComplex* pffted_rowsignal, cufftComplex* pcarrCD_Out
	, cufftComplex* pcarrTemp, const unsigned int LEnChunk, const unsigned int N_p, fdmt_type_* d_parr_fdmt_inp
	, const unsigned int IMaxDT, const /*long*/ double VAl_practicalD, const float VAlD_max, const float VAlFmin
	, const float VAlFmax, void* pAuxBuff_fdmt, const int IDeltaT, cufftHandle plan0
	, cufftHandle plan1, cufftComplex* pAuxBuff);

void fncCoherentDedispersion_gpu(cufftComplex* pcarrCD_Out, cufftComplex* pcarrffted_rowsignal
	, const unsigned int LEnChunk, const /*long*/ double VAl_practicalD, const float VAlFmin, const float VAlFmax
	, cufftHandle  plan_backward, cufftComplex* pAuxBuff);

void fncSTFT_gpu(cufftComplex* pcarrOut, cufftComplex* pRawSignalCur, const unsigned int LEnChunk, int block_size
	, cufftHandle plan_short, cufftComplex* pAuxBuff);

__global__
void kernel_Aux(float* psum, float* psumSq, cufftComplex* d_mtrxSig
	, unsigned int len);

__global__ void transpose(cufftComplex* input, cufftComplex* output, int width, int height);

__global__
void kernel_create_arr_fdmt_inp(fdmt_type_* d_parr_fdmt_inp, cufftComplex* d_mtrxPower, unsigned int len
	, const float val_mean, const float val_stdDev);

__global__ void kernel_OneSM_Mean_and_Disp(float* d_arrMeans, float* d_arrDisps, int len
	, float* pvalMean, float* pvalDisp);

__global__ void calcPowerMtrx_and_RowMeans_and_Disps
(fdmt_type_* d_parr_fdmt_inp_, cufftComplex* pcarrTemp_, const int NRows, const int NCols
	, float* d_arrSumMean, float* d_arrSumMeanSquared);

__global__ void kernel_normalize_array(fdmt_type_* pAuxBuff, const unsigned int len
	, float* pmean, float* pdev);

void calc_fdmt_inp(fdmt_type_* d_parr_fdmt_inp, cufftComplex* pcarrTemp, unsigned int nRows, unsigned int nCols
	, void* pAuxBuff);

__global__ void kernel_calcAuxArray(cufftComplex* pAuxBuff, cufftComplex* pcarrffted_rowsignal
	, const unsigned int LEnChunk, const /*long*/ double step, const /*long*/ double VAl_practicalD, const double fmin
	, const double fmax);

__global__
void scaling_kernel(cufftComplex* data, int element_count, float scale)	;

int malloc_for_4_complex_arrays(cufftComplex** ppffted_rowsignal, cufftComplex** ppcarrTemp
	, cufftComplex** ppcarrCD_Out, cufftComplex** ppcarrBuff, const unsigned int LEnChunk);

__global__
void fncSignalDetection_gpu(fdmt_type_* parr_fdmt_out, fdmt_type_* parrImNormalize, const unsigned int qCols
	, const unsigned int len, fdmt_type_* pmaxElement, unsigned int* argmaxRow, unsigned int* argmaxCol);

inline int calcThreadsForMean_and_Disp(unsigned const int nCols)
{
	int k = std::log(nCols) / std::log(2.0);
	k = ((1 << k) > nCols) ? k + 1 : k;
	return 1 << std::min(k, 10);
}

bool createOutImageForFixedNumberChunk_gpu(fdmt_type_** parr_fdmt_out, int* pargmaxRow, int* pargmaxCol, fdmt_type_* pvalSNR
	, fdmt_type_** pparrOutSubImage, int* piQuantRowsPartImage, CStreamParams* pStreamPars, const int numChunk
	, const float VAlCoherent_d);

void cutQuadraticSubImage(fdmt_type_** pparrOutImage, int* piQuantRowsOutImage, fdmt_type_* InpImage
	, const int QInpImageRows, const int QInpImageCols, const int NUmCentralElemRow, const int NUmCentralElemCol);

//--------------------------------------------------
__global__ void kernel_Temp0(double* arr_t4
	, const unsigned int LEnChunk, const /*long*/ double step, const /*long*/ double VAl_practicalD, const double fmin
	, const double fmax);

__global__ void kernel_Temp1(cufftComplex* pAuxBuff, cufftComplex* pcarrffted_rowsignal, double* arr_t4
	, const unsigned int LEnChunk, const /*long*/ double step, const /*long*/ double VAl_practicalD, const double fmin
	, const double fmax);

__device__ double my_modf(double val, double* intpart);

