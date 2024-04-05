#pragma once
#include "stdio.h" 
#include "TelescopeHeader.h"
#include <cufft.h>
#include <math.h>
#include "Constants.h"
#include <vector>
//enum EN_channelOrder { STRAIGHT, INVERTED };

#define MAX_PATH_LENGTH 1000

extern const unsigned long long TOtal_GPU_Bytes;
class CTelescopeHeader;
class COutChunkHeader;
class CFragment;

class CGuppiSession
{
public:
	~CGuppiSession();
	CGuppiSession();
	CGuppiSession(const  CGuppiSession& R);
	CGuppiSession& operator=(const CGuppiSession& R);
	CGuppiSession(const char* strGuppiPath, const char* strOutPutPath, const float t_p
		, const double d_max, const float sigma_bound, const int length_sum_wnd);
	//--------------------------------------------------------	
	
	char m_strGuppiPath[MAX_PATH_LENGTH];
	char m_strOutPutPath[MAX_PATH_LENGTH];
	
	CTelescopeHeader m_header;
	float m_pulse_length;
	double m_d_max;
	float m_sigma_bound;
	int m_length_sum_wnd;
	std::vector<COutChunkHeader>* m_pvctSuccessHeaders;
	
	//----------------------------------------------------------
	int calcQuantRemainBlocks(FILE* rbFile, unsigned long long* pilength);

	int launch();	

	bool analyzeChunk(const COutChunkHeader outChunkHeader, CFragment* pFRg);

	bool navigateToBlock(FILE* rbFile, const int IBlockNum);

	size_t  downloadChunk(FILE* rb_file, char* d_parrInput, const long long QUantDownloadingBytes);	

};

__global__
void unpackInput(cufftComplex* pcmparrRawSignalCur, inp_type_* d_parrInput, const int  lenChunk
	, const int  nchan, const int  npol);


