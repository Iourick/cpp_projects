#pragma once
#include "stdio.h"
#include <vector>
#include <cufft.h>
#include "Constants.h"
#include <complex> 
#include "FdmtGpu.cuh"
#include "Chunk_gpu.cuh"
//#include "ChunkB.h"
#define TILE_DIM 32
using namespace std;

class COutChunkHeader;
class CFragment;
class CFdmtU;
class CTelescopeHeader;
 
class CChunk_v1_gpu : public  CChunk_gpu
{
public:
	~CChunk_v1_gpu();
	CChunk_v1_gpu();
	CChunk_v1_gpu(const  CChunk_v1_gpu& R);
	CChunk_v1_gpu& operator=(const CChunk_v1_gpu& R);
	CChunk_v1_gpu(
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
	
	cufftComplex* m_pd_arr_dc;
	

	virtual void compute_chirp_channel();

	virtual void  elementWiseMult(cufftComplex* d_arrOut, cufftComplex* d_arrInp0, int  idm);

};
__global__
void kernel_create_arr_dc(cufftComplex* parr_dc, double* parrcoh_dm, double* parr_freqs_chan, double* parr_bin_freqs, double* parr_taper
	, int ndm, int nchan, int len_sft, int mbin);

__global__ void transpose_(float* odata, float* idata, int width, int height);



__global__ 	
void  transpose_unpadd_intensity(float* fbuf, float* arin, int nfft, int noverlap_per_channel
	, int mbin_adjusted, const int nsub, const int nchan, int mbin);

__global__ 	void  transpose_unpadd_intensity__(float* fbuf, float* arin, int nfft, int noverlap_per_channel
	, int mbin_adjusted, const int nsub, const int nchan, int mbin);

//__global__	void  transpose_unpadd_gpu(float* fbuf, float* arin, int nfft, int noverlap_per_channel
//	, int mbin_adjusted, const int nsub, const int nchan, int mbin);









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




