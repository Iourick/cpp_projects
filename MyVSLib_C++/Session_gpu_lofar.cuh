#pragma once
#include "stdio.h" 
#include "TelescopeHeader.h"
#include <cufft.h>
#include <math.h>
#include "Constants.h"
#include <vector>
#include "Session_gpu.cuh"



class CSession_gpu_lofar :public CSession_gpu
{
public:
	CSession_gpu_lofar();
	CSession_gpu_lofar(const  CSession_gpu_lofar& R);
	CSession_gpu_lofar& operator=(const CSession_gpu_lofar& R);
	CSession_gpu_lofar(const char* strlofarPath, const char* strOutPutPath, const float t_p
		, const double d_max, const float sigma_bound, const int length_sum_wnd);
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
		, inp_type_* d_parrInput, cufftComplex* pcmparrRawSignalCur);

	virtual void rewindFilePos(FILE** prb_File, const int  QUantTotalChannelBytes);

	virtual bool closeFileReadingStream(FILE**& prb_File);

};
struct header_h5 read_h5_header(char* fname);

__global__
void unpackInput_L(cufftComplex* pcmparrRawSignalCur, inp_type_* d_parrInput, const int  lenChunk
	, const int  NChan, const int  npol);