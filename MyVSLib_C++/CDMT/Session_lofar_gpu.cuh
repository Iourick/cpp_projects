#pragma once
#include "Session_lofar.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cufft.h>

class CSession_lofar;
class CChunkB;
class CSession_lofar_gpu :public CSession_lofar
{
public:
	CSession_lofar_gpu();
	CSession_lofar_gpu(const  CSession_lofar_gpu& R);
	CSession_lofar_gpu& operator=(const CSession_lofar_gpu& R);
	CSession_lofar_gpu(const char* strGuppiPath, const char* strOutPutPath, const float t_p
		, const double d_min, const double d_max, const float sigma_bound, const int length_sum_wnd, const int nbin, const int nfft);
	
	virtual bool unpack_chunk(const long long lenChunk, const int j, inp_type_* d_parrInput, void* pcmparrRawSignalCur);

	virtual bool allocateInputMemory(void** parrInput, const int QUantDownloadingBytesForChunk, void** pcmparrRawSignalCur
		, const int QUantChunkComplexNumbers);

	virtual void createChunk(CChunkB** ppchunk
        , const float Fmin
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
        , const float tsamp);

	virtual void freeInputMemory(void* parrInput, void* pcmparrRawSignalCur);

    virtual size_t download_chunk(FILE** prb_File, char* d_parrInput, const long long QUantDownloadingBytes);

};


__global__
void unpackInput_lofar(cufftComplex* pcmparrRawSignalCur, inp_type_* d_parrInput, const int NPolPh, const int LenChunk, const int nbin, const int  Noverlap);

//inline int calcThreadsForMean_and_Disp(unsigned const int nCols)
//{
//    int k = std::log(nCols) / std::log(2.0);
//    k = ((1 << k) > nCols) ? k + 1 : k;
//    return 1 << std::min(k, 10);
//};