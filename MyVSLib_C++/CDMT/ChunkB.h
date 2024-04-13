#pragma once
//#include "stdio.h"
#include <vector>
#include <math.h>
#include "Constants.h"

class COutChunkHeader;
class CChunkB
{
public:
	
	CChunkB();
	CChunkB(const  CChunkB& R);
	CChunkB& operator=(const CChunkB& R);
	CChunkB(
		const float Fmin
		, const float Fmax
		, const int npol
		, const int nchan
		//, const unsigned int lenChunk
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

	//--------------------------------------------------------
	float m_Fmin;

	float m_Fmax;
	// NPOL: Number of samples per time step—4 corresponds to
	 //dual - polarization complex data.
	unsigned int m_npol;

	//The number of frequency channels contained
	//within the file (=OBSNCHAN).
	unsigned int m_nchan;

	// length of time series of each channel
	
	//unsigned m_lenChunk;

	unsigned int m_len_sft;

	float m_tsamp;

	int m_Chunk_id;

	int m_Block_id;

	double m_d_max;

	double m_d_min;

	int m_ncoherent;

	float m_sigma_bound;

	int m_length_sum_wnd;
	int m_nbin;
	int m_nfft;
	int m_noverlap;
	//-------------------------------------------------------------------------
	virtual bool process(void* pcmparrRawSignalCur
		, std::vector<COutChunkHeader>* pvctSuccessHeaders, std::vector<std::vector<float>>* pvecImg);


	inline int get_noverlap_per_channel()
	{
		return m_noverlap / m_len_sft;
	};

	inline int get_mbin_adjusted()
	{
		return get_mbin() - 2 * get_noverlap_per_channel();
	}

	inline int get_mbin()
	{
		return  m_nbin / m_len_sft;
	}

	inline int get_msamp()
	{
		return m_nfft * get_mbin_adjusted();
	}


//
//
//	static long long calcLenChunk_(CTelescopeHeader header, const int nsft
//		, const float pulse_length, const float d_max);

	void set_chunkid(const int nC);

	void set_blockid(const int nC);

	//int get_coherent_dms();
	

	static void cutQuadraticFragment(float* parrFragment, float* parrInpImage, int* piRowBegin, int* piColBegin
		, const int QInpImageRows, const int QInpImageCols, const int NUmTargetRow, const int NUmTargetCol);
};
//

inline int calcThreadsForMean_and_Disp(unsigned const int nCols)
{
	int k = log(nCols) / log(2.0);
	k = ((1 << k) > nCols) ? k + 1 : k;
	return 1 << std::min(k, 10);
};





void windowization(float* d_fdmt_normalized, const int Rows, const int Cols, const int width, float* parrImage);



