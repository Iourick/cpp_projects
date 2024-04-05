#include "Chunk_cpu.h"
//
//#include <vector>
//#include "OutChunkHeader.h"
//
//#include "Constants.h"
//
//
//
#include <chrono>
//
//
//
//
#include <complex>
//
////#include "Fragment.h"
#include "npy.hpp"
//#include "TelescopeHeader.h"

#include <fftw3.h>





//extern const unsigned long long TOtal_GPU_Bytes = (long long)free_bytes;


//
CChunk_cpu::CChunk_cpu() :CChunkB()
{

}
//-----------------------------------------------------------

CChunk_cpu::CChunk_cpu(const  CChunk_cpu& R) :CChunkB(R)
{
}
//-------------------------------------------------------------------

CChunk_cpu& CChunk_cpu::operator=(const CChunk_cpu& R)
{
	if (this == &R)
	{
		return *this;
	}
	CChunkB:: operator= (R);
	return *this;
}
//------------------------------------------------------------------
CChunk_cpu::CChunk_cpu(
	const float Fmin
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
	, const int noverlap) : CChunkB(Fmin
		,  Fmax
		, npol
		, nchan
		,  lenChunk
		,len_sft
		,  Block_id
		,  Chunk_id
		,  d_max
		,  d_min
		,  ncoherent
		,  sigma_bound
		, length_sum_wnd
		,  nbin
		,  nfft
		,  noverlap)
{
}

//
//
////---------------------------------------------------
bool CChunk_cpu::process(void* pcmparrRawSignalCur
	, std::vector<COutChunkHeader>* pvctSuccessHeaders)
{
	
	return true;
}

bool CChunk_cpu::try0()
{
	return true;
 }
//-----------------------------------------------------------------
//
//long long CChunk_cpu::calcLenChunk_(CTelescopeHeader header, const int nsft
//	, const float pulse_length, const float d_max)
//{
//	//const int nchan_actual = nsft * header.m_nchan;
//
//	//long long len = 0;
//	//for (len = 1 << 9; len < 1 << 30; len <<= 1)
//	//{
//	//	CFdmtU fdmt(
//	//		header.m_centfreq - header.m_chanBW * header.m_nchan / 2.
//	//		, header.m_centfreq + header.m_chanBW * header.m_nchan / 2.
//	//		, nchan_actual
//	//		, len
//	//		, pulse_length
//	//		, d_max
//	//		, nsft
//	//	);
//	//	long long size0 = fdmt.calcSizeAuxBuff_fdmt_();
//	//	long long size_fdmt_inp = fdmt.calc_size_input();
//	//	long long size_fdmt_out = fdmt.calc_size_output();
//	//	long long size_fdmt_norm = size_fdmt_out;
//	//	long long irest = header.m_nchan * header.m_npol * header.m_nbits / 8 // input buff
//	//		+ header.m_nchan * header.m_npol / 2 * sizeof(cufftComplex)
//	//		+ 3 * header.m_nchan * header.m_npol * sizeof(cufftComplex) / 2
//	//		+ 2 * header.m_nchan * sizeof(float);
//	//	irest *= len;
//
//	//	long long rez = size0 + size_fdmt_inp + size_fdmt_out + size_fdmt_norm + irest;
//	//	if (rez > 0.98 * TOtal_GPU_Bytes)
//	//	{
//	//		return len / 2;
//	//	}
//
//	//}
//	return -1;
//}










