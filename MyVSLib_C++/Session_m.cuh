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

	long long calcLenChunk(const int n_p);

	

	bool analyzeChunk(const COutChunkHeader outChunkHeader, CFragment* pFRg);

	bool navigateToBlock(const int IBlockNum);

	static inline int calc_len_sft(const float chanBW, const double t_p)
	{
		return (t_p > 2. / (chanBW * 1.0E6)) ? pow(2, ceil(log2(t_p * chanBW * 1.0E6))) : 1;
	}

	static inline unsigned int calc_MaxDT(const float val_fmin_MHz, const float val_fmax_MHz, const float time_resolution_sec
		, const float val_DM_Max)
	{
		return (int)(4148.8 * val_DM_Max * (1. / (val_fmin_MHz * val_fmin_MHz) - 1.
			/ (val_fmax_MHz * val_fmax_MHz)) / time_resolution_sec);
	}

};


