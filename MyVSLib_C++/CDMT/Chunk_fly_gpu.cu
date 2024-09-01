#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Chunk_fly_gpu.cuh"

#include <vector>
#include "OutChunkHeader.h"

#include "Constants.h"


#include <chrono>

#include "helper_functions.h"
#include "helper_cuda.h"
#include <math_functions.h>

#include <complex>

#include "npy.hpp"
#include "TelescopeHeader.h"
extern cudaError_t cudaStatus0;	

	extern const unsigned long long TOtal_GPU_Bytes;;

	
    #define BLOCK_DIM 32//16
	CChunk_fly_gpu::~CChunk_fly_gpu()
	{		
	}
	//-----------------------------------------------------------
	CChunk_fly_gpu::CChunk_fly_gpu() :CChunk_gpu()
	{			
	}
	//-----------------------------------------------------------

	CChunk_fly_gpu::CChunk_fly_gpu(const  CChunk_fly_gpu& R) :CChunk_gpu(R)
	{
		
	}
	//-------------------------------------------------------------------

	CChunk_fly_gpu& CChunk_fly_gpu::operator=(const CChunk_fly_gpu& R)
	{
		if (this == &R)
		{
			return *this;
		}
		CChunk_gpu:: operator= (R);
		
		return *this;
	}
	//------------------------------------------------------------------
	CChunk_fly_gpu::CChunk_fly_gpu(
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
	}

//------------------------------------------------------------------------------------------
// INPUT:
// d_arrInp - raw data array after straight FFT
// idm 
// OUTPUT:
//d_arrOut
void  CChunk_fly_gpu::elementWiseMult(cufftComplex* d_arrOut, cufftComplex* d_arrInp	, int  idm)
{	
	double bw = m_Fmax - m_Fmin;
	int mbin = get_mbin();
	double bw_sub = bw / m_nchan;  // m_nchan - channels quantity
	double bw_chan = bw_sub / m_len_sft;// m_len_sft = 32,  

	//INPUT:
	// d_arrInp - raw data array after straight FFT
	// m_pd_arrcoh_dm[idm] - current dm
	// m_nchan - quant of channels
	//  m_len_sft - quantity of "subchannelization" (=32)
	// mbin-  bins in time domain  for fdmt input
	// m_nfft -fft quantity
	// m_npol - quantity of polarizations, = 2 OR 4
	// bw_sub, bw_chan - are defined above
	const dim3 block_Size2(256, 1, 1);
	const dim3 gridSize2((mbin + block_Size2.x - 1) / block_Size2.x, m_len_sft, m_nchan);
	kernel_el_wise_mult_onthe_fly << < gridSize2, block_Size2 >> >
		(d_arrOut, d_arrInp, &m_pd_arrcoh_dm[idm], m_nchan, m_len_sft, mbin, m_nfft, m_npol, m_Fmin, bw_sub, bw_chan);	
	 
}
//-------------------------------------------------------------------------------------------
// element wise multiplication on shifted phase-array 
__global__
void kernel_el_wise_mult_onthe_fly(cufftComplex* parr_Out, cufftComplex* parr_Inp, double* pdm
	, int nchan, int len_sft, int mbin, int nfft, int npol, double Fmin, double bw_sub, double bw_chan)
{
	int ibin = blockIdx.x * blockDim.x + threadIdx.x;
	if (ibin >= mbin)
	{
		return;
	}
	int ichan = blockIdx.z;
	int isft = blockIdx.y;

	float temp0 = Fmin + bw_sub * (0.5 + ichan) + bw_chan * (isft - len_sft / 2.0 + 0.5);
	int i0 = ichan * len_sft * mbin + isft * mbin;

	//
	double temp = -0.5 * bw_chan + (ibin + 0.5) * bw_chan / mbin;
	double bin_freqs = temp;
	double taper = 1.0 / sqrt(1.0 + pow(temp / (0.47 * bw_chan), 80));

	double temp1 = bin_freqs / temp0;
	double phase_delay = ((*pdm) * temp1 * temp1 / (temp0 + bin_freqs) * 4.148808e9);
	double val_prD_int = 0;
	double t = -modf(phase_delay, &val_prD_int) * 2.0;
	double val_x = 0.0, val_y = 0.;

	sincospi(t, &val_y, &val_x);

	int nbin = mbin * len_sft;
	int num_el = i0 + ibin;
	int i1 = num_el / nbin;
	int ibin1 = num_el % nbin;
	ibin1 = (ibin1 + nbin / 2) % nbin;

	cufftComplex dc;	
	dc.x = float(val_x * taper / (double(nbin)));
	dc.y = float(val_y * taper / (double(nbin)));

	int ind = i1 * nbin + ibin1;
	int stride = nbin * nchan;
	for (int i = 0; i < nfft * npol / 2; ++i)
	{
		parr_Out[ind] = cmpMult(parr_Inp[ind], dc);
		ind += stride;
	}
}

//-----------------------------------------------------------------------
cufftComplex cmpMult(cufftComplex& a, cufftComplex&b)
{
	cufftComplex r;
	r.x = a.x * b.x - a.y * b.y;
	r.y = a.x * b.y + a.y * b.x;
	return r;
}









