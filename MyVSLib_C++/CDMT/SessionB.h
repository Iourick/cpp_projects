#pragma once
#include "stdio.h" 
#include "TelescopeHeader.h"
#include <math.h>
#include "Constants.h"
#include <vector>
#include "ChunkB.h"



#define MAX_PATH_LENGTH 1000

extern const unsigned long long TOtal_GPU_Bytes;
class CTelescopeHeader;
class COutChunkHeader;
//class CFragment;
class CFdmtU;
class CChunk_cpu;
 
enum TYPE_OF_INP_FORMAT
{
	GUPPI
	, FLOFAR
};
enum TYPE_OF_PROCESSOR
{
	CPU
	, GPU
};

class CSessionB
{
public:
	virtual~CSessionB();
	CSessionB();
	CSessionB(const  CSessionB& R);
	CSessionB& operator=(const CSessionB& R);
	CSessionB(const char* strGuppiPath, const char* strOutPutPath, const float t_p
		, const double d_min, const double d_max, const float sigma_bound, const int length_sum_wnd
		, const int nbin, const int nfft);
	//--------------------------------------------------------	
	
	char m_strInpPath[MAX_PATH_LENGTH];
	char m_strOutPutPath[MAX_PATH_LENGTH];
	
	CTelescopeHeader m_header;
	float m_pulse_length;
	double m_d_max;
	double m_d_min;
	float m_sigma_bound;
	int m_length_sum_wnd;	
	int m_nbin;
	int m_nfft;

	std::vector<COutChunkHeader>* m_pvctSuccessHeaders;
	
	//----------------------------------------------------------
	virtual int calcQuantBlocks( unsigned long long* pilength);

	virtual bool openFileReadingStream(FILE**& prb_File);

	virtual bool closeFileReadingStream(FILE**& prb_File);

	int launch(std::vector<std::vector<float>>* pvecImg, int* pmsamp)	;	

	virtual bool navigateToBlock(FILE* rbFile, const int IBlockNum);

	virtual size_t  download_chunk(FILE** rb_file, char* d_parrInput, const long long QUantDownloadingBytes);

	int get_optimal_overlap(const int nsft);

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

	virtual bool allocateInputMemory(void** parrInput, const int QUantDownloadingBytesForChunk, void** pcmparrRawSignalCur
		, const int QUantChunkComplexNumbers);	

	/*static long long _calcLenChunk_(CTelescopeHeader header, const int nsft
		, const float pulse_length, const float d_max);*/

	virtual bool createCurrentTelescopeHeader(FILE**prb_File);

	

	virtual bool unpack_chunk(const long long lenChunk, const int j
		, inp_type_* d_parrInput, void* pcmparrRawSignalCur);

	virtual void rewindFilePos(FILE** prb_File, const int  QUantTotalChannelBytes);

	

	int  get_coherent_dms();

	virtual void freeInputMemory(void* parrInput, void* pcmparrRawSignalCur);

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

	virtual void  shift_file_pos(FILE** prb_File, const int IShift);

	virtual void shift_file_ptr_to_last_pos(FILE** prb_File, const long long quant_bytes_per_polarization);

	inline int calc_ChunkBytes(const int LenChunk)
	{
		return LenChunk * m_header.m_nchan / 8 * m_header.m_npol * m_header.m_nbits;
	}

	inline int  calc_TotalChannelBytes()
	{
		return m_header.m_nblocksize * m_header.m_nbits / 8 / m_header.m_nchan;
	}

	inline int calc_ChunkComplexNumbers()
	{
		return   m_nfft * m_header.m_nchan * m_nbin * (m_header.m_npol / 2);
	}

	inline int calc_ChunkPolarizationComplexNumbers()
	{
		return   m_nfft * m_header.m_nchan * m_nbin ;
	}

	virtual size_t  calc_ShiftingBytes(unsigned int QUantChunkBytes);	
		
};




