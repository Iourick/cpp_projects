#pragma once
#include "Session_guppi.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cufft.h>

class CSession_guppi;
class CChunkB;
class CSession_guppi_gpu :public CSession_guppi
{
public:
	CSession_guppi_gpu();
	CSession_guppi_gpu(const  CSession_guppi_gpu& R);
	CSession_guppi_gpu& operator=(const CSession_guppi_gpu& R);
	CSession_guppi_gpu(const char* strGuppiPath, const char* strOutPutPath, const float t_p
		, const double d_min, const double d_max, const float sigma_bound, const int length_sum_wnd, const int nbin, const int nfft);
	
	virtual bool  unpack_chunk(const long long lenChunk, const int j, inp_type_* d_parrInput, void* pcmparrRawSignalCur);

	virtual bool allocateInputMemory(void** parrInput, const int QUantDownloadingBytesForChunk, void** pcmparrRawSignalCur
		, const int QUantChunkComplexNumbers);

    virtual void createChunk(CChunkB** ppchunk
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


