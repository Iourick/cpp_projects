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
class CFdmtU;
class CChunk;
 
enum TYPE_OF_INP_FORMAT
{
	 GUPPI
	,FLOFAR
};

class CSession_gpu
{
public:
	~CSession_gpu();
	CSession_gpu();
	CSession_gpu(const  CSession_gpu& R);
	CSession_gpu& operator=(const CSession_gpu& R);
	CSession_gpu(const char* strGuppiPath, const char* strOutPutPath, const float t_p
		, const double d_max, const float sigma_bound, const int length_sum_wnd);
	//--------------------------------------------------------	
	
	char m_strInpPath[MAX_PATH_LENGTH];
	char m_strOutPutPath[MAX_PATH_LENGTH];
	
	CTelescopeHeader m_header;
	float m_pulse_length;
	double m_d_max;
	float m_sigma_bound;
	int m_length_sum_wnd;
	std::vector<COutChunkHeader>* m_pvctSuccessHeaders;
	
	//----------------------------------------------------------
	virtual int calcQuantBlocks( unsigned long long* pilength);

	virtual bool openFileReadingStream(FILE**& prb_File);

	virtual bool closeFileReadingStream(FILE**& prb_File);

	int launch();	

	bool analyzeChunk(const COutChunkHeader outChunkHeader, CFragment* pFRg);

	bool navigateToBlock(FILE* rbFile, const int IBlockNum);

	

	bool do_plan_and_memAlloc( int* pLenChunk
		, cufftHandle* pplan0, cufftHandle* pplan1, CFdmtU* pfdmt, char** d_pparrInput, cufftComplex** ppcmparrRawSignalCur
		, void** ppAuxBuff_fdmt, fdmt_type_** d_parrfdmt_norm
		, cufftComplex** ppcarrTemp
		, cufftComplex** ppcarrCD_Out
		, cufftComplex** ppcarrBuff, char** ppInpOutBuffFdmt, CChunk** ppChunk);

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

	void plan_and_memAlloc( int* pLenChunk
		, cufftHandle* pplan0, cufftHandle* pplan1, CFdmtU* pfdmt, char** d_pparrInput, cufftComplex** ppcmparrRawSignalCur
		, void** ppAuxBuff_fdmt, fdmt_type_** d_parrfdmt_norm
		, cufftComplex** ppcarrTemp
		, cufftComplex** ppcarrCD_Out
		, cufftComplex** ppcarrBuff, char** ppInpOutBuffFdmt, CChunk** ppChunk);

	static long long _calcLenChunk_(CTelescopeHeader header, const int nsft
		, const float pulse_length, const float d_max);

	virtual bool createCurrentTelescopeHeader(FILE**prb_File);

	virtual  void download_and_unpack_chunk(FILE** prb_File, const long long lenChunk, const int j
		, inp_type_* d_parrInput, cufftComplex* pcmparrRawSignalCur);

	virtual void rewindFilePos(FILE** prb_File, const int  QUantTotalChannelBytes);
};



__global__
void unpackInput(cufftComplex* pcmparrRawSignalCur, inp_type_* d_parrInput, const int  lenChunk
	, const int  nchan, const int  npol);


