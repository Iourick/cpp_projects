#pragma once
#include "stdio.h"
#include <vector>
#include <cufft.h>
#include "Constants.h"
#include <complex> 
#include "FdmtGpu.cuh"
#include "Chunk_gpu.cuh"

#define TILE_DIM 32
using namespace std;

class COutChunkHeader;

class CFdmtU;
class CTelescopeHeader;
 
class CChunk_fly_gpu : public  CChunk_gpu
{
public:
	~CChunk_fly_gpu();
	CChunk_fly_gpu();
	CChunk_fly_gpu(const  CChunk_fly_gpu& R);
	CChunk_fly_gpu& operator=(const CChunk_fly_gpu& R);
	CChunk_fly_gpu(
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

	virtual void  elementWiseMult(cufftComplex* d_arrOut, cufftComplex* d_arrInp0, int  idm);

};

__device__ cufftComplex cmpMult(cufftComplex& a, cufftComplex& b);

__global__
void kernel_el_wise_mult_onthe_fly(cufftComplex* parr_Out, cufftComplex* parr_Inp, double* pdm
	, int nchan, int len_sft, int mbin, int nfft, int npol, double Fmin, double bw_sub, double bw_chan);














//inline int calcThreadsForMean_and_Disp(unsigned const int nCols)
//{
//	int k = std::log(nCols) / std::log(2.0);
//	k = ((1 << k) > nCols) ? k + 1 : k;
//	return 1 << std::min(k, 10);
//};

//__global__ 
//void normalize_and_clean(fdmt_type_* parrOut, float* d_arr, const int NRows, const int NCols
//	, float* pmean, float* pstd, float* d_arrRowDisp, float* pmeanDisp, float* pstdDisp);




//void windowization(float* d_fdmt_normalized, const int Rows, const int Cols, const int width, float* parrImage);
//
//__global__
//void fdmt_normalization(fdmt_type_* d_arr, fdmt_type_* d_norm, const int lenChunk, float* d_pOutArray);

//__global__
//void multiTransp_kernel(float* output, const int height, const int width, float* input);




