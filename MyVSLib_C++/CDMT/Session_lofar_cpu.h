#pragma once
#include "stdio.h" 
#include "TelescopeHeader.h"

#include <math.h>
#include "Constants.h"
#include <vector>
#include "Session_lofar.h"
#include <fftw3.h>


class CChunkB;
class CSession_lofar;
class CSession_lofar_cpu :public CSession_lofar
{
public:
	CSession_lofar_cpu();
	CSession_lofar_cpu(const  CSession_lofar_cpu& R);
	CSession_lofar_cpu& operator=(const CSession_lofar_cpu& R);
	CSession_lofar_cpu(const char* strGuppiPath, const char* strOutPutPath, const float t_p
		, const double d_min, const double d_max, const float sigma_bound, const int length_sum_wnd, const int nbin, const int nfft);
	

	virtual bool unpack_chunk(const long long LenChunk, const int Noverlap
		, inp_type_* d_parrInput, void* pcmparrRawSignalCur);

	

	virtual bool allocateInputMemory(void** parrInput, const int QUantDownloadingBytesForChunk, void** pcmparrRawSignalCur
		, const int QUantChunkComplexNumbers);

	virtual void  createChunk(CChunkB** ppchunk
		, const float Fmin
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
		, const float tsamp);

	virtual void freeInputMemory(void* parrInput, void* pcmparrRawSignalCur);

};


