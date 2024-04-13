#include "ChunkB.h"
#include <chrono>
#include <complex>
//#include "npy.hpp"

#ifdef _WIN32 // Windows

#include <Windows.h>

void emitSound(int frequency, int duration) {
	Beep(frequency, duration);
}

#else // Linux

#include <cmath>
#include <alsa/asoundlib.h>

void emitSound(int frequency, int duration) {
	int rate = 44100; // Sampling rate
	snd_pcm_t* handle;
	snd_pcm_open(&handle, "default", SND_PCM_STREAM_PLAYBACK, 0);
	snd_pcm_set_params(handle, SND_PCM_FORMAT_S16_LE, SND_PCM_ACCESS_RW_INTERLEAVED, 1, rate, 1, 500000);

	short buf[rate * duration];

	for (int i = 0; i < rate * duration; i++) {
		int sample = 32760 * sin(2 * M_PI * frequency * i / rate);
		buf[i] = sample;
	}

	snd_pcm_writei(handle, buf, rate * duration);
	snd_pcm_close(handle);
}

#endif   



//extern const unsigned long long TOtal_GPU_Bytes = (long long)free_bytes;

// timing variables:
  // fdmt time
//long long iFdmt_time = 0;
//// read && transform data time
//long long  iReadTransform_time = 0;
//// fft time
//long long  iFFT_time = 0;
//// detection time
//long long  iMeanDisp_time = 0;
//// detection time
//long long  iNormalize_time = 0;
//// total time
//long long  iTotal_time = 0;

//
CChunkB::CChunkB()
{
	m_Fmin = 0;
	m_Fmax = 0;
	m_npol = 0;

	m_nchan = 0;	
	m_len_sft = 0;
	m_Block_id = 0;
	m_Chunk_id = -1;

	m_d_max = 0.;
	m_d_min = 0.;
	m_ncoherent = 0;
	m_sigma_bound = 10.;
	m_length_sum_wnd = 10;

	m_nbin = 0;
	m_nfft = 0;
	m_noverlap = 0;
	m_tsamp = 0.;
}
//-----------------------------------------------------------

CChunkB::CChunkB(const  CChunkB& R)
{
	m_Fmin = R.m_Fmin;
	m_Fmax = R.m_Fmax;
	m_npol = R.m_npol;
	m_nchan = R.m_nchan;	
	m_len_sft = R.m_len_sft;
	m_Chunk_id = R.m_Chunk_id;
	m_Block_id = R.m_Block_id;
	m_d_max = R.m_d_max;
	m_d_min = R.m_d_min;
	m_sigma_bound = R.m_sigma_bound;
	m_length_sum_wnd = R.m_length_sum_wnd;
	m_nbin = R.m_nbin;
	m_nfft = R.m_nfft;
	m_noverlap = R.m_noverlap;
	m_ncoherent = R.m_ncoherent;
	m_tsamp = R.m_tsamp;
}
//-------------------------------------------------------------------

CChunkB& CChunkB::operator=(const CChunkB& R)
{
	if (this == &R)
	{
		return *this;
	}
	m_Fmin = R.m_Fmin;
	m_Fmax = R.m_Fmax;
	m_npol = R.m_npol;
	m_nchan = R.m_nchan;	
	m_len_sft = R.m_len_sft;
	m_Chunk_id = R.m_Chunk_id;
	m_Block_id = R.m_Block_id;
	m_d_max = R.m_d_max;
	m_d_min = R.m_d_min;
	m_sigma_bound = R.m_sigma_bound;
	m_length_sum_wnd = R.m_length_sum_wnd;
	m_nbin = R.m_nbin;
	m_nfft = R.m_nfft;
	m_noverlap = R.m_noverlap;
	m_ncoherent = R.m_ncoherent;
	m_tsamp = R.m_tsamp;
	return *this;
}
//------------------------------------------------------------------
CChunkB::CChunkB(
	const float Fmin
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
	,const float tsamp
)
{
	m_Fmin = Fmin;
	m_Fmax = Fmax;
	m_npol = npol;

	m_nchan = nchan;	
	m_len_sft = len_sft;
	m_Block_id = Block_id;
	m_Chunk_id = Chunk_id;

	m_d_max = d_max;
	m_d_min = d_min;
	m_ncoherent = ncoherent;
	m_sigma_bound = sigma_bound;
	m_length_sum_wnd = length_sum_wnd;

	m_nbin = nbin;
	m_nfft = nfft;
	m_noverlap = noverlap;
	m_tsamp = tsamp;
}

//
//
////---------------------------------------------------
bool CChunkB::process(void* pcmparrRawSignalCur
	, std::vector<COutChunkHeader>* pvctSuccessHeaders, std::vector<std::vector<float>>* pvecImg)
{
	
	return true;
}
////-------------------------------------------------
//
//int  CChunkB::get_coherent_dms()
//{
//	// Compute the coherent DMs for the FDMT algorithm.
//	
//	
//	float    t_d = 4.148808e3 * m_Fmax * (1.0 / (m_Fmin * m_Fmin) - 1.0 / (m_Fmax * m_Fmax));
//	int irez = ceil(t_d * m_header.m_tresolution / (m_pulse_length * m_pulse_length));
//	return irez;
//}

//-----------------------------------------------------------------
//
//long long CChunkB::calcLenChunk_(CTelescopeHeader header, const int nsft
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

//
//--------------------------------------
void CChunkB::set_chunkid(const int nC)
{
	m_Chunk_id = nC;
}
//--------------------------------------
void CChunkB::set_blockid(const int nC)
{
	m_Block_id = nC;
}
//-------------------------------------------------------------------


//------------------------------------------------------
void CChunkB::cutQuadraticFragment(float* parrFragment, float* parrInpImage, int* piRowBegin, int* piColBegin
	, const int QInpImageRows, const int QInpImageCols, const int NUmTargetRow, const int NUmTargetCol)
{
	if (QInpImageRows < QInpImageCols)
	{
		int numPart = NUmTargetCol / QInpImageRows;
		int numColStart = numPart * QInpImageRows;
		for (int i = 0; i < QInpImageRows; ++i)
		{
			memcpy(&parrFragment[i * QInpImageRows], &parrInpImage[i * QInpImageCols + numColStart], QInpImageRows * sizeof(float));
		}
		*piColBegin = numColStart;
		*piRowBegin = 0;
		return;
	}
	int numPart = NUmTargetRow / QInpImageRows;
	int numStart = numPart * QInpImageCols;
	memcpy(parrFragment, &parrInpImage[numStart], QInpImageCols * QInpImageCols * sizeof(float));
	*piRowBegin = numPart;
	*piColBegin = 0;
}







