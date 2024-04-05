#pragma once
//#include "stdio.h"
#include <vector>
#include <math.h>
//
#include "Constants.h"
#include "ChunkB.h"
//
//
//using namespace std;
//
class COutChunkHeader;
class CChunkB;
//class CFragment;
//
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
		, const unsigned int lenChunk
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
	);

	//-------------------------------------------------------------------------
	virtual bool process(void* pcmparrRawSignalCur
		, std::vector<COutChunkHeader>* pvctSuccessHeaders);

	virtual bool try0();
};
//







