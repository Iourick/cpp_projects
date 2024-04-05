#pragma once
#include "stdio.h" 
#include "TelescopeHeader.h"
#include <cufft.h>
#include <math.h>
#include "Constants.h"
#include <vector>
#include "Session_gpu.cuh"



class CSession_gpu_guppi :public CSession_gpu
{
public:	
	CSession_gpu_guppi();
	CSession_gpu_guppi(const  CSession_gpu_guppi& R);
	CSession_gpu_guppi& operator=(const CSession_gpu_guppi& R);
	CSession_gpu_guppi(const char* strGuppiPath, const char* strOutPutPath, const float t_p
		, const double d_max, const float sigma_bound, const int length_sum_wnd);
	//--------------------------------------------------------	



	//----------------------------------------------------------
	int calcQuantBlocks(unsigned long long* pilength);

	virtual float fncTest(float val);

	virtual bool readTelescopeHeader(FILE* r_File
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

	virtual bool openFileReadingStream(FILE**& prb_File);

	virtual bool createCurrentTelescopeHeader(FILE** prb_File);

	virtual void download_and_unpack_chunk(FILE** prb_File, const long long lenChunk, const int j
		, inp_type_* d_parrInput, cufftComplex* pcmparrRawSignalCur);

	virtual void rewindFilePos(FILE** prb_File, const int  QUantTotalChannelBytes);

	virtual bool closeFileReadingStream(FILE**& prb_File);

	size_t downloadChunk(FILE* rb_File, char* d_parrInput, const long long QUantDownloadingBytes);

};