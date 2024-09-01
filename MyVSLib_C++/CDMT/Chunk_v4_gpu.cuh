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
class CFdmtU;
class CTelescopeHeader;
 
class CChunk_v4_gpu : public  CChunk_gpu
{
public:
	~CChunk_v4_gpu();
	CChunk_v4_gpu();
	CChunk_v4_gpu(const  CChunk_v4_gpu& R);
	CChunk_v4_gpu& operator=(const CChunk_v4_gpu& R);
	CChunk_v4_gpu(
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
	// buffer for keeping dc array for each dm step
	cufftComplex* m_pd_buff_dc;

	// taper array
	double* m_pdarr_taper;

	// one auxillary array
	double* m_pdarr_bin_freqs;
	

	virtual void compute_chirp_channel();

	virtual void  elementWiseMult(cufftComplex* d_arrOut, cufftComplex* d_arrInp0, int  idm);

	void compute_current_chirp(int idm);

};
__global__
void kernel_create_current_chirp_(cufftComplex* parr_dc, double* pdm, double* parr_bin_freqs, double* parr_taper
	, int nchan, int len_sft, int mbin, double Fmin, double bw_sub, double bw_chan);







