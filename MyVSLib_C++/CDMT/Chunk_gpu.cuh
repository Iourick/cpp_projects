#pragma once
#include "stdio.h"
#include <vector>
#include <cufft.h>
#include "Constants.h"
#include <complex> 
#include "FdmtGpu.cuh"
#include "ChunkB.h"
#define TILE_DIM 32
using namespace std;

class COutChunkHeader;
class CFragment;
class CFdmtU;
class CTelescopeHeader;
 
class CChunk_gpu : public CChunkB
{
public:
	~CChunk_gpu();
	CChunk_gpu();
	CChunk_gpu(const  CChunk_gpu& R);
	CChunk_gpu& operator=(const CChunk_gpu& R);
	CChunk_gpu(
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
		, const float tsamp);
	//---------------------------------------------------------------------------------------
	double* m_pd_arrcoh_dm;

	cufftComplex* m_pd_arr_dc;

	cufftHandle m_fftPlanForward;

	cufftHandle m_fftPlanInverse;

	CFdmtGpu m_Fdmt;
	

	//-------------------------------------------------------------------------
	virtual bool process(void* pcmparrRawSignalCur
		, std::vector<COutChunkHeader>* pvctSuccessHeaders, std::vector<std::vector<float>>* pvecImg);	

	void calc_fdmt_inp(fdmt_type_* d_parr_fdmt_inp, cufftComplex* pcarrTemp
		, float* pAuxBuff);		

	static long long calcLenChunk_(CTelescopeHeader header, const int nsft
		, const float pulse_length, const float d_max);

	void set_chunkid(const int nC);

	void set_blockid(const int nC);

	static void preparations_and_memoryAllocations(CTelescopeHeader header
		, const float pulse_length
		, const float d_max
		, const float sigma_bound
		, const int length_sum_wnd
		, int* pLenChunk
		, cufftHandle* pplan0, cufftHandle* pplan1, CFdmtU* pfdmt, char** d_pparrInput, cufftComplex** ppcmparrRawSignalCur
		, void** ppAuxBuff_fdmt, fdmt_type_** d_parrfdmt_norm, cufftComplex** ppcarrTemp, cufftComplex** ppcarrCD_Out
		, cufftComplex** ppcarrBuff, char** ppInpOutBuffFdmt,  CChunk_gpu** ppChunk);

	void compute_chirp_channel();

	void create_fft_plans();

};
__global__
void kernel_create_arr_freqs_chan(double* d_parr_freqs_chan, int len_sft, double bw_chan, double  Fmin, double bw_sub);

__global__
void kernel_create_arr_bin_freqs_and_taper(double* d_parr_bin_freqs, double* d_parr_taper, double  bw_chan, int mbin);

__global__
void kernel_create_arr_dc(cufftComplex* parr_dc, double* parrcoh_dm, double* parr_freqs_chan, double* parr_bin_freqs, double* parr_taper
	, int ndm, int nchan, int len_sft, int mbin);

__global__ 
void roll_rows_and_normalize_kernel(cufftComplex* arr_rez, cufftComplex* arr, int rows, int cols, int shift);

__global__ 
void  element_wise_cufftComplex_mult_kernel(cufftComplex* d_arrOut, cufftComplex* d_arrInp0, cufftComplex* d_arrInp1
	, int npol, int nfft, int dim2);

__global__
void element_wise_cufftComplex_mult_kernel(cufftComplex* data, long long element_count, float scale);

__global__ void  divide_cufftComplex_array_kernel(cufftComplex* d_arr, int len, float val);

__global__
void  transpose_unpadd(cufftComplex* fbuf, cufftComplex* arin, int nfft, int noverlap_per_channel
	, int mbin_adjusted, const int nchan, const int nlen_sft, int mbin);

__global__
void scaling_kernel(cufftComplex* data, long long element_count, float scale);

__device__
float fnc_norm2(cufftComplex* pcarr);

__global__
void calcPartSum_kernel(float* d_parr_out, const int lenChunk, const int npol_physical, cufftComplex* d_parr_inp);

__global__
void calcMultiTransposition_kernel(fdmt_type_* output, const int height, const int width, fdmt_type_* input);

__global__
void calcPowerMtrx_kernel(float* output, const int height, const int width, const int npol, cufftComplex* input);

//inline int calcThreadsForMean_and_Disp(unsigned const int nCols)
//{
//	int k = std::log(nCols) / std::log(2.0);
//	k = ((1 << k) > nCols) ? k + 1 : k;
//	return 1 << std::min(k, 10);
//};

__global__ 
void normalize_and_clean(fdmt_type_* parrOut, float* d_arr, const int NRows, const int NCols
	, float* pmean, float* pstd, float* d_arrRowDisp, float* pmeanDisp, float* pstdDisp);




void windowization(float* d_fdmt_normalized, const int Rows, const int Cols, const int width, float* parrImage);

__global__
void fdmt_normalization(fdmt_type_* d_arr, fdmt_type_* d_norm, const int lenChunk, float* d_pOutArray);

__global__
void multiTransp_kernel(float* output, const int height, const int width, float* input);




