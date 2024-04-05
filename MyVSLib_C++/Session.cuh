#pragma once
#include "stdio.h" 
#include "TelescopeHeader.h"
#include <cufft.h>
#include <math.h>
#include "Constants.h"
#include <vector>

#define MAX_PATH_LENGTH 1000

extern const unsigned long long TOtal_GPU_Bytes;
class CTelescopeHeader;
class COutChunkHeader;
class CFragment;

class CSession
{
public:
	~CSession();
	CSession();
	CSession(const  CSession& R);
	CSession& operator=(const CSession& R);
	CSession(const char* strGuppiPath, const char* strOutPutPath, const float t_p
		, const double d_max, const float sigma_bound, const int length_sum_wnd);
	//--------------------------------------------------------	
	FILE* m_rbFile;
	FILE* m_wb_file;
	char m_strGuppiPath[MAX_PATH_LENGTH];
	char m_strOutPutPath[MAX_PATH_LENGTH];
	
	CTelescopeHeader m_header;
	float m_t_p;
	double m_d_max;
	float m_sigma_bound;
	int m_length_sum_wnd;
	std::vector<COutChunkHeader>* m_pvctSuccessHeaders;
	//----------------------------------------------------------
	int calcQuantRemainBlocks(unsigned long long* pilength);

	int launch();

	void writeReport();

	inline int calc_len_sft(const float chanBW)
	{
		return (m_t_p > 2. / (chanBW * 1.0E6)) ? pow(2, ceil(log2(m_t_p * chanBW * 1.0E6))) : 1;
	}

	long long CSession::calcLenChunk(const int n_p);

	static bool read_outputlogfile_line(const char* pstrPassLog
		, const int NUmLine
		, int* pnumBlock
		, int* pnumChunk
		, int* pn_fdmtRows
		, int* n_fdmtCols
		, int* psucRow
		, int* psucCol
		, int* pwidth
		, float* pcohDisp
		, float* snr
	);

	bool analyzeChunk(const COutChunkHeader outChunkHeader, CFragment* pFRg);

	bool navigateToBlock(const int IBlockNum);

};


