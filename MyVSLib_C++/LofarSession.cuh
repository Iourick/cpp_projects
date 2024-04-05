#pragma once
#include "stdio.h" 
#include "TelescopeHeader.h"
#include <cufft.h>
#include <math.h>
#include "Constants.h"
#include <vector>
#include "FdmtU.cuh"

#define MAX_PATH_LENGTH 1000

extern const unsigned long long TOtal_GPU_Bytes;
class CTelescopeHeader;
class COutChunkHeader;
class CFragment;
class CChunk;

class CLofarSession
{
public:
	~CLofarSession();
	CLofarSession();
	CLofarSession(const  CLofarSession& R);
	CLofarSession& operator=(const CLofarSession& R);
	CLofarSession(const char* strInputPath, const char* strOutPutPath, const float t_p
		, const double d_max, const float sigma_bound, const int length_sum_wnd);
	//--------------------------------------------------------	
	
	char m_strInputPath[MAX_PATH_LENGTH];
	char m_strOutPutPath[MAX_PATH_LENGTH];
	
	CTelescopeHeader m_header;

	//m_pulse_length - time resolution 
	float m_pulse_length;
	double m_d_max;
	float m_sigma_bound;
	int m_length_sum_wnd;
	std::vector<COutChunkHeader>* m_pvctSuccessHeaders;
	//----------------------------------------------------------
	

	int launch();
	
		

	bool analyzeChunk(const COutChunkHeader outChunkHeader, CFragment* pFRg);

	
	
};

struct header_h5 read_h5_header(char* fname);



__global__
void unpackInput(cufftComplex* pcmparrRawSignalCur, inp_type_* d_parrInput, const int  lenChunk
	, const int  NChan, const int  npol);




