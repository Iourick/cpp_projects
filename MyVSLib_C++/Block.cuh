#pragma once
#include "stdio.h"
#include <vector>
#include <cufft.h>
#include "Constants.h"
#define TILE_DIM 32
using namespace std;

enum EN_channelOrder { STRAIGHT, INVERTED };

class COutChunkHeader;
class CFragment;
struct structOutDetection;
class CBlock
{
public:
	//~CBlock();
	CBlock();
	CBlock(const  CBlock& R);
	CBlock& operator=(const CBlock& R);
	CBlock(
		const float Fmin
		, const float Fmax
		, const int npol
		, const int nblocksize
		, const int nchan
		, const unsigned int lenChunk
		, const unsigned int len_sft
		, const int bulk_id
		, const int nbits
		, const bool bOrderStraight
		, const double d_max
		, const float sigma_bound
		, const int length_sum_wnd
	);
	//--------------------------------------------------------
	float m_Fmin;
	float m_Fmax;
	// NPOL: Number of samples per time step—4 corresponds to
	 //dual - polarization complex data.
	unsigned int m_npol;
	// BLOCSIZE: The size of the following raw data segment, in
		//bytes.Does not include padding bytes from Direct I / O.Can
		//be expressed as 2 x NPOL x NTIME x NCHAN x NBITS /8
	unsigned long long m_nblocksize;
	//The number of frequency channels contained
	//within the file (=OBSNCHAN).
	unsigned int m_nchan;

	// length of time series of each channel
	unsigned int m_lenChunk;

	unsigned int m_len_sft;

	//std::vector<COutChunkHeader>* m_pvctSuccessHeaders;

	int m_block_id;

	int m_nbits;

	EN_channelOrder m_enchannelOrder;

	double m_d_max;

	float m_sigma_bound;

	int m_length_sum_wnd;
	

	//-------------------------------------------------------------------------
	int process(FILE* rb_file, std::vector<COutChunkHeader>* pvctSuccessHeaders);

	size_t  downloadChunk(FILE* rb_file, char* d_parrInput, const long long QUantDownloadingBytes);

	bool fncChunkProcessing_gpu(cufftComplex* pcmparrRawSignalCur
		, void* pAuxBuff_fdmt		
		, cufftComplex* pcarrTemp
		, cufftComplex* pcarrCD_Out
		, cufftComplex* pcarrBuff
		, float* pAuxBuff_flt, fdmt_type_* d_arrfdmt_norm
		, const int IDeltaT, cufftHandle plan0, cufftHandle plan1
		, structOutDetection* pstructOut
		, float* pcoherentDedisp);

	int calcFDMT_Out_gpu(fdmt_type_* parr_fdmt_out, cufftComplex* pffted_rowsignal, cufftComplex* pcarrCD_Out
		, cufftComplex* pcarrTemp, fdmt_type_* d_parr_fdmt_inp
		, const unsigned int IMaxDT, const  double VAl_practicalD
		, void* pAuxBuff_fdmt, const int IDeltaT, cufftHandle plan0, cufftHandle plan1, cufftComplex* pAuxBuff);

	void fncCD_Multiband_gpu(cufftComplex* pcarrCD_Out, cufftComplex* pcarrffted_rowsignal
		, const  double VAl_practicalD, cufftHandle  plan, cufftComplex* pAuxBuff);

	void fncSTFT_gpu(cufftComplex* pcarrOut, cufftComplex* pRawSignalCur
		, cufftHandle plan_short, cufftComplex* pAuxBuff);

	void calc_fdmt_inp(fdmt_type_* d_parr_fdmt_inp, cufftComplex* pcarrTemp
		, float* pAuxBuff);

	

	bool detailedChunkProcessing(FILE* rb_file, const COutChunkHeader outChunkHeader, CFragment* pFRg);

	bool detailedChunkAnalysus_gpu(cufftComplex* pcmparrRawSignalCur
		, void* pAuxBuff_fdmt
		, cufftComplex* pcarrTemp
		, cufftComplex* pcarrCD_Out
		, cufftComplex* pcarrBuff
		, float* pAuxBuff_flt, fdmt_type_* d_arrfdmt_norm
		, const int IDeltaT, cufftHandle plan0, cufftHandle plan1
		, const COutChunkHeader outChunkHeader
		, CFragment* pFRgtor
	);

};

__global__
void unpackInput(cufftComplex* pcmparrRawSignalCur, inp_type_* d_parrInput, const int  lenChunk
	, const int  nchan, const int  npol);


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

inline int calcThreadsForMean_and_Disp(unsigned const int nCols)
{
	int k = std::log(nCols) / std::log(2.0);
	k = ((1 << k) > nCols) ? k + 1 : k;
	return 1 << std::min(k, 10);
};

__global__ 
void normalize_and_clean(fdmt_type_* parrOut, float* d_arr, const int NRows, const int NCols
	, float* pmean, float* pstd, float* d_arrRowDisp, float* pmeanDisp, float* pstdDisp);

void cutQuadraticFragment(float* parrFragment, float* parrInpImage, int* piRowBegin, int* piColBegin
	, const int QInpImageRows, const int QInpImageCols, const int NUmTargetRow, const int NUmTargetCol);


void windowization(float* d_fdmt_normalized, const int Rows, const int Cols, const int width, float* parrImage);

__global__
void fdmt_normalization(fdmt_type_* d_arr, fdmt_type_* d_norm, const int lenChunk, float* d_pOutArray);

__global__
void multiTransp_kernel(float* output, const int height, const int width, float* input);

__global__ void kernel_ElementWiseMult(cufftComplex* pAuxBuff, cufftComplex* pcarrffted_rowsignal
	, const unsigned int LEnChunk, const unsigned int n_pol_phys
	, const  double VAl_practicalD, const double Fmin
	, const double Fmax);

//__global__
//void fncSignalDetection_gpu(fdmt_type_* parr_fdmt_out, fdmt_type_* parrImNormalize, const unsigned int qCols
//	, const unsigned int len, fdmt_type_* pmaxElement, unsigned int* argmaxRow, unsigned int* argmaxCol);



