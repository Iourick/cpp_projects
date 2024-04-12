#pragma once
#include "SessionB.h"
#include "stdio.h" 
#include "TelescopeHeader.h"

#include <math.h>
#include "Constants.h"
#include <vector>
#include <fftw3.h>

class CSession_lofar :public CSessionB
{
public:
	CSession_lofar();
	CSession_lofar(const  CSession_lofar& R);
	CSession_lofar& operator=(const CSession_lofar& R);
	CSession_lofar(const char* strGuppiPath, const char* strOutPutPath, const float t_p
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

	virtual size_t download_chunk(FILE** rb_File, char* d_parrInput, const long long QUantDownloadingBytes);

	virtual bool unpack_chunk(const long long LenChunk, const int Noverlap
		, inp_type_* d_parrInput, void* pcmparrRawSignalCur);

	virtual void rewindFilePos(FILE** prb_File, const int  QUantTotalChannelBytes);

	virtual bool closeFileReadingStream(FILE**& prb_File);

	

	virtual bool allocateInputMemory(void** parrInput, const int QUantDownloadingBytesForChunk, void** pcmparrRawSignalCur
		, const int QUantChunkComplexNumbers);

	virtual void  createChunk(CChunkB** ppchunk
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

	virtual void  shift_file_pos(FILE** prb_File, const int IShift);

	virtual size_t  calc_ShiftingBytes(unsigned int QUantChunkBytes);

};
struct header_h5 read_h5_header(char* fname);