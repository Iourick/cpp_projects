#pragma once

#include "ChunkB.h"
#include "FdmtCpu.h"
class COutChunkHeader;
class CChunkB;
class CFdmtCpu;
#include <fftw3.h>
#include <complex> 
#define STR_WIS_LEN 10000

//class CTelescopeHeader;

class CChunk_cpu:public CChunkB
{
public:
	
	CChunk_cpu();
	CChunk_cpu(const  CChunk_cpu& R);
	CChunk_cpu& operator=(const CChunk_cpu& R);
	CChunk_cpu(
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
		, const float tsamp
	);

	std::vector<std::complex<float>> m_dc_Vector;

	CFdmtCpu m_fdmt;
	

	char m_str_wis_forw[STR_WIS_LEN];

	char m_str_wis_back[STR_WIS_LEN];
	
	//-------------------------------------------------------------------------
	virtual bool process(void* pcmparrRawSignalCur
		, std::vector<COutChunkHeader>* pvctSuccessHeaders, std::vector<std::vector<float>>* pvecImg);

	void compute_chirp_channel(std::vector<std::complex<float>>* parr_dc, const  std::vector<double>* parr_coh_dm);

	static void fnc_roll_and_normalize_ffted(fftwf_complex* pcmparr_ffted, const int NRows, const int NCols);

	static void fnc_element_wise_mult(std::complex<float>* arr0, fftwf_complex* arr1, fftwf_complex* arrOut, const int LEn);

	void  transpose_unpadd(fftwf_complex* arin, fftwf_complex* arout);

     void fnc_roll_ffted(fftwf_complex* pcmparr_ffted, const int NRows, const int NCols);


	template <typename T>
	static void roll_(T* arr, const int lenarr, const int ishift);

	void fnc_dedisperse(float* parr_wfall, const float dm, const float  tsamp);

	//bool detect_signal(fdmt_type_* arr, fdmt_type_* norm, const int Rows, const int  Cols, structOutDetection* pstrOut);

	//void detect_signal_in_row(fdmt_type_* arr, fdmt_type_* norm, const int  Cols, const int  NumRow, structOutDetection* pstrOut);
};







