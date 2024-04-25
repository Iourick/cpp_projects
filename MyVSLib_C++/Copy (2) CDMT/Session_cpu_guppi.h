#pragma once
#include "stdio.h" 
#include "TelescopeHeader.h"

#include <math.h>
#include "Constants.h"
#include <vector>
#include "SessionB.h"
#include <fftw3.h>


class CChunkB;
class CSession_cpu_guppi :public CSessionB
{
public:
	CSession_cpu_guppi();
	CSession_cpu_guppi(const  CSession_cpu_guppi& R);
	CSession_cpu_guppi& operator=(const CSession_cpu_guppi& R);
	CSession_cpu_guppi(const char* strGuppiPath, const char* strOutPutPath, const float t_p
		, const double d_min, const double d_max, const float sigma_bound, const int length_sum_wnd, const int nbin, const int nfft);
	//--------------------------------------------------------	
	virtual bool openFileReadingStream(FILE**& prb_File);


	//----------------------------------------------------------
	virtual int calcQuantBlocks(unsigned long long* pilength);

	virtual bool readTelescopeHeader(FILE* r_file
		, int* nbits
		, float* chanBW
		, int* npol
		, bool* bdirectIO
		, float* centfreq
		, int* nchan
		, float* obsBW
		, long long* nblocksize
		, EN_telescope* TELESCOP
		, float* tresolution
	);

	virtual bool createCurrentTelescopeHeader(FILE** prb_File);

	virtual void download_and_unpack_chunk(FILE** prb_File, const long long lenChunk, const int j
		, inp_type_* d_parrInput, void* pcmparrRawSignalCur);

	virtual void rewindFilePos(FILE** prb_File, const int  QUantTotalChannelBytes);

	virtual bool closeFileReadingStream(FILE**& prb_File);

	

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
		, const int noverlap);

	virtual void freeInputMemory(void* parrInput, void* pcmparrRawSignalCur);

	void unpack_input_cpu_guppi(fftwf_complex* pcmparrRawSignalCur, inp_type_* d_parrInput, const int  lenChunk, const int nchan, const int npol);
};