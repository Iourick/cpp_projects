#include "Chunk_cpu.h"

#include <chrono>

#include <cmath>
#include <fftw3.h>
#include "npy.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
CChunk_cpu::CChunk_cpu() :CChunkB()
{

}
//-----------------------------------------------------------

CChunk_cpu::CChunk_cpu(const  CChunk_cpu& R) :CChunkB(R)
{
	if (!(R.m_dc_Vector.empty()))
	{
		m_dc_Vector = R.m_dc_Vector;
	}
	if (!(R.m_coh_dm_Vector.empty()))
	{
		m_coh_dm_Vector = R.m_coh_dm_Vector;
	}
	strcpy(m_str_wis_forw, R.m_str_wis_forw);
	strcpy(m_str_wis_back, R.m_str_wis_back);
}
//-------------------------------------------------------------------

CChunk_cpu& CChunk_cpu::operator=(const CChunk_cpu& R)
{
	if (this == &R)
	{
		return *this;
	}
	CChunkB:: operator= (R);
	if (!(R.m_dc_Vector.empty()))
	{
		m_dc_Vector = R.m_dc_Vector;
	}

	if (!(R.m_coh_dm_Vector.empty()))
	{
		m_coh_dm_Vector = R.m_coh_dm_Vector;
	}

	strcpy(m_str_wis_forw, R.m_str_wis_forw);
	strcpy(m_str_wis_back, R.m_str_wis_back);
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
	, const int noverlap
	, const float tsamp) : CChunkB(Fmin
		,  Fmax
		, npol
		, nchan
		,  lenChunk
		, len_sft
		,  Block_id
		,  Chunk_id
		,  d_max
		,  d_min
		,  ncoherent
		,  sigma_bound
		,  length_sum_wnd
		,  nbin
		,  nfft
		,  noverlap
		,  tsamp)
{
	// 1. create coh_dm array
	const double coh_dm_step = m_d_max / m_ncoherent;
	const int ndm = (m_d_max - m_d_min) / coh_dm_step;
	m_coh_dm_Vector.resize(ndm);
	for (int i = 0; i < ndm; ++i) {
		m_coh_dm_Vector[i] = m_d_min + i * coh_dm_step;
	}
	// 1!

	//2.create dc vector
	m_dc_Vector.resize(ndm * m_nchan * m_nbin);
	compute_chirp_channel(&m_dc_Vector, &m_coh_dm_Vector);
	// 2!

	//3. create wisdom strings for FFT
	  // 3.1 forward FFT
	int rank = 1, n = nbin, howmany = nchan * npol / 2 * nfft;
	int inembed = n;
	int istride = 1;
	int  idist = n;
	int onembed = n;
	int ostride = 1;
	int odist = n;
	int length = n * howmany;
	fftwf_complex* in = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * length);
	fftwf_complex* out = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * length);
	fftwf_plan plan = NULL;
	plan = fftwf_plan_many_dft(1, &n, howmany,
		in, &n, 1, n, out, &n, 1, n, FFTW_FORWARD, FFTW_MEASURE);

	strcpy(m_str_wis_forw , fftwf_export_wisdom_to_string());
	fftwf_destroy_plan(plan);

	 n = nbin / len_sft, howmany = nchan * npol / 2 * nfft *len_sft;

	 plan = fftwf_plan_many_dft(1, &n, howmany,
		 in, &n, 1, n, out, &n, 1, n, FFTW_BACKWARD, FFTW_MEASURE);

	 strcpy(m_str_wis_back, fftwf_export_wisdom_to_string());
	 fftwf_destroy_plan(plan);

	 fftwf_free(in);
	 fftwf_free(out);

}


////---------------------------------------------------
bool CChunk_cpu::process(void* pcmparrRawSignalCur	, std::vector<COutChunkHeader>* pvctSuccessHeaders)
{	
	// 0. 
	const int mbin = get_mbin();
	const int noverlap_per_channel = get_noverlap_per_channel();
	const int mbin_adjusted = get_mbin_adjusted();
	const int msamp = get_msamp();
	const int mchan = m_nchan * m_len_sft;
	// 0!
	
	// 1. Forward FFT execution
	fftwf_cleanup();
	fftw_import_wisdom_from_string(m_str_wis_forw);
	int  howmany = m_nchan * m_npol / 2 * m_nfft;
	fftwf_plan plan  = fftwf_plan_many_dft(1, &m_nbin, howmany,
		(fftwf_complex*)pcmparrRawSignalCur, &m_nbin, 1, m_nbin, (fftwf_complex*)pcmparrRawSignalCur, &m_nbin, 1, m_nbin, FFTW_FORWARD, FFTW_ESTIMATE);
	fftwf_execute(plan);
	fftwf_destroy_plan(plan);
	//1!	
	
	//2. roll and normalize ffted signals
	fnc_roll_and_normalize_ffted((fftwf_complex*)pcmparrRawSignalCur, m_nchan * m_npol / 2 * m_nfft, m_nbin);
	//2!

	

	// 3. memory allocation for buffer
	fftwf_complex*  parrElemWiseMulted = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * m_nchan * m_npol / 2 * m_nfft * m_nbin);
	//fbuf = np.zeros((msamp, nsub, nchan), dtype = np.complex64)
	//	nfft, nsub, nchan, mbin = cp.shape
	
	fftwf_complex* fbuf = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * msamp* m_nchan * m_len_sft  * m_npol / 2 );
	float* parr_wfall = (float*)malloc(sizeof(float) * msamp * m_nchan * m_len_sft);
	// 4. main loop
	fftwf_complex* parr_ffted = (fftwf_complex*)pcmparrRawSignalCur;
	for (int idm = 0; idm < m_coh_dm_Vector.size(); ++idm)
	{
		// 5. elementwise multiplication
		std::complex<float>* pnt = m_dc_Vector.data();
		pnt += m_nchan * m_nbin * idm;
		for (int i = 0; i < m_npol / 2; ++i)
		{
			for (int j = 0; j < m_nfft; ++j)
			{
				fnc_element_wise_mult(pnt, parr_ffted + i * m_nfft * m_nchan * m_nbin +  j * m_nchan * m_nbin, parrElemWiseMulted + i * m_nfft * m_nchan * m_nbin + j * m_nchan * m_nbin, m_nchan * m_nbin);
			}			
		}
		// 5!

		//6. roll and normalize ffted signals
		fnc_roll_ffted(parrElemWiseMulted, m_nchan * m_npol / 2 * m_len_sft* m_nfft, m_nbin/ m_len_sft);
		//6!

		
		
		// 7.
		fftwf_cleanup();
		fftw_import_wisdom_from_string(m_str_wis_back);
		int n = m_nbin / m_len_sft, howmany = m_nchan * m_npol / 2 * m_nfft * m_len_sft;
		fftwf_plan plan = fftwf_plan_many_dft(1, &n, howmany,
			parrElemWiseMulted, &n, 1, n, parrElemWiseMulted, &n, 1, n, FFTW_BACKWARD, FFTW_ESTIMATE);
		fftwf_execute(plan);
		fftwf_destroy_plan(plan);
		// 7!
		
		// 8. normalization of parrElemWiseMulted,
		float coef = 1.0f / ((float)n);
		for (int i = 0; i < m_nchan * m_nfft * m_nbin * m_npol / 2;++i)
		{
			parrElemWiseMulted[i][0] *= coef;
			parrElemWiseMulted[i][1] *= coef;
		}		
		// 8	
		
		// 9.	unpadding and transposition
		transpose_unpadd(parrElemWiseMulted,  fbuf);
		transpose_unpadd(parrElemWiseMulted + m_nchan * m_nfft * m_nbin,  fbuf + msamp * m_nchan * m_len_sft);
		// 9!
		
		//10. calc intensity matrix already transposed
		
		memset(parr_wfall, 0, sizeof(float) * msamp * m_nchan * m_len_sft);
		const int NRows = m_nchan * m_len_sft;
		const int NCols = msamp;
		const int NEl = NRows * NCols;
		for (int i = 0; i < NRows; ++i)
		{
			for (int j = 0; j < NCols; ++j)
			{
				for (int ipol = 0; ipol < m_npol / 2; ++ipol)
				{
					parr_wfall[i * NCols + j] += fbuf[ipol * NEl + j * NRows + i][0] * fbuf[ipol * NEl + j * NRows + i][0] + fbuf[ipol * NEl + j * NRows + i][1] * fbuf[ipol * NEl + j * NRows + i][1];
				}
			}
		}
		// 9!
		
		
		fnc_dedisperse(parr_wfall, m_coh_dm_Vector[idm], m_tsamp * m_len_sft);
		//---------------- 

		//std::complex<float>* pcp = (std::complex<float> *)fbuf;
		std::vector<float> vec(parr_wfall, parr_wfall + NEl);

		// Accessing elements
		std::array<long unsigned, 1> leshape127{ NEl };

		npy::SaveArrayAsNumpy("pcmparrRawSignalCur_c.npy", false, leshape127.size(), leshape127.data(), vec);
		int ii = 0;
		//--------------------
		
	}

	fftwf_free(parrElemWiseMulted);
	fftwf_free(fbuf);
	free(parr_wfall);
	return true;
}
//--------------------------------------------------------------
template <typename T>
void CChunk_cpu::roll_(T* arr, const int lenarr, const int ishift)
{
	if (ishift == 0)
	{
		return;
	}
	T* arr0 = (T*)malloc(ishift * sizeof(T));
	T* arr1 = (T*)malloc((lenarr - ishift) * sizeof(T));
	memcpy(arr0, arr + lenarr - 1 - ishift, ishift * sizeof(T));
	memcpy(arr1, arr, (lenarr - ishift) * sizeof(T));
	memcpy(arr, arr0, ishift * sizeof(T));
	memcpy(arr + ishift, arr1, (lenarr - ishift) * sizeof(T));
	free(arr0);
	free(arr1);
}
template void CChunk_cpu::roll_<int>(int* arr, const int lenarr, const int ishift);
template void CChunk_cpu::roll_<float>(float* arr, const int lenarr, const int ishift);

void CChunk_cpu::fnc_dedisperse(float* parr_wfall, const float dm, const float  tsamp)
{
	const int msamp = get_msamp();
	const int nchans = m_len_sft * m_nchan;
	double f_min = (double)m_Fmin;
	double f_max = (double)m_Fmax;
	double foff = (f_max - f_min) / nchans;
	//double *arr_chan_edge_freqs = (double*)malloc(nchans * sizeof(double));
	for (int i = 0; i < nchans; ++i)
	{
		double temp  = (double )i * foff + f_min;
		double temp1 = 4.148808e3 * dm * (1.0 / (f_min * f_min) - 1.0 / (temp * temp));
		int ishift = round(temp1 / tsamp);
		roll_(parr_wfall + (nchans - 1 - i) * msamp, msamp, ishift);
	}

	//free(arr_chan_edge_freqs);
}
//
//def dedisperse(
//	waterfall: np.ndarray,
//	dm : float,
//	f_min : float,
//	f_max : float,
//	nchans : int,
//	tsamp : float,
//	) :
//	new_ar = waterfall.copy()
//	foff = (f_max - f_min) / nchans
//	chan_edge_freqs = np.arange(nchans) * foff + f_min
//	chan_edge_freqs = chan_edge_freqs[:: - 1]
//	delays = 4.148808e3 * dm * ((f_min * *-2) - (chan_edge_freqs * *-2))
//	shift = (delays / tsamp).round().astype("int32")
//	for ichan in range(waterfall.shape[0]) :
//		new_ar[ichan] = np.roll(waterfall[ichan], shift[ichan])
//		return new_ar

// INPUT:
// OUTPUT:
//parr_dc
void CChunk_cpu::compute_chirp_channel(std::vector<std::complex<float>>*parr_dc,const  std::vector<float>  *parr_coh_dm)
{
	// 1 preparations
	double bw = m_Fmax - m_Fmin;
	int mbin = get_mbin();
	double bw_sub = bw / m_nchan;
	double bw_chan = bw_sub / m_len_sft;
	int ndm = parr_coh_dm->size();
	//1!

	// 2.arr_freqs_sub computation
	std::vector<double>arr_freqs_sub(m_nchan);
	for (int i = 0; i < m_nchan; ++i)
	{
		arr_freqs_sub[i] = m_Fmin + bw_sub * (0.5 + i);
	}
	// 2!

	// 3. arr_freqs_chan computation
	std::vector<double>arr_freqs_chan(m_nchan* m_len_sft);
	std::vector<double>arr_temp( m_len_sft);
	for (int i = 0; i < m_len_sft; ++i)
	{
		arr_temp[i] = bw_chan * ((double)i - m_len_sft / 2.0 + 0.5);
	}
	for (int i = 0; i < m_nchan; ++i)
	{
		for (int j = 0; j < m_len_sft; ++j)
		{
			arr_freqs_chan[i * m_len_sft + j] = arr_freqs_sub[i] + arr_temp[j];
		}
	}
	// 3!

	// 4. arr_bin_freqs computation
	std::vector<double>arr_bin_freqs(mbin);
	for (int i = 0; i < mbin; ++i)
	{
		arr_bin_freqs[i] = -0.5 * bw_chan + (i + 0.5) * bw_chan / mbin;
	}
	// 4!

	// 5 phase_delay
	std::vector<double> arr_phase_delay(ndm * m_nchan * m_len_sft * mbin);
	for (int idm = 0; idm < ndm; ++idm)
		for (int ichan = 0; ichan < m_nchan; ++ichan)
		{
			for (int isft = 0; isft < m_len_sft; ++isft)
			{
				double temp0 = arr_freqs_chan[ichan * m_len_sft + isft];
				#pragma omp parallel // 
				{
					for (int ibin = 0; ibin < mbin; ++ibin)
					{
						double temp1 = arr_bin_freqs[ibin] / temp0;
						arr_phase_delay[idm * m_nchan * m_len_sft * mbin + ichan * m_len_sft * mbin + isft * mbin + ibin]
							= (*parr_coh_dm)[idm] * temp1 * temp1 / (temp0 + arr_bin_freqs[ibin]) * 4.148808e9;
					}
				}// ! #pragma omp parallel
			}
		}
	// then element wise mult
	std::vector<double>taper(mbin);
	
	for (int i = 0; i < mbin; ++i)
	{
		taper[i] = 1.0 / sqrt(1.0 + pow(arr_bin_freqs[i] / (0.47 * bw_chan), 80) );
	}
	std::complex<double>cmpi(0.0, 1.0);
	for (int idm = 0; idm < ndm; ++idm)
		for (int ichan = 0; ichan < m_nchan; ++ichan)
		{
			for (int isft = 0; isft < m_len_sft; ++isft)
			{
				#pragma omp parallel // 
				{
					for (int ibin = 0; ibin < mbin; ++ibin)
					{
						(*parr_dc)[idm * m_nchan * m_len_sft * mbin + ichan * m_len_sft * mbin + isft * mbin + ibin]
							= std::exp(-2.0 * cmpi * M_PI * arr_phase_delay[idm * m_nchan * m_len_sft * mbin + ichan * m_len_sft * mbin + isft * mbin + ibin]) * taper[ibin];

					}
				} // ! #pragma omp parallel
			}
		}
	

	/*std::array<long unsigned, 4> leshape127 {ndm, m_nchan,  m_len_sft,mbin };
	npy::SaveArrayAsNumpy("chirp_c.npy", false, leshape127.size(), leshape127.data(), *parr_dc);	
	int ii = 0;*/
}
//-----------------------------------------------------------------------------------------------------
void CChunk_cpu::fnc_roll_and_normalize_ffted(fftwf_complex* pcmparr_ffted, const int NRows, const int NCols)
{
	float val = 1.0f / ((float)NCols);
	fftwf_complex* arrtemp = pcmparr_ffted;
	for (int i = 0; i < NRows; ++i)
	{
		for (int j = 0; j < NCols / 2; ++j)
		{
			fftwf_complex temp;
			temp[0] = arrtemp[j][0];
			temp[1] = arrtemp[j][1];
			arrtemp[j][0] = arrtemp[NCols / 2 + j][0] * val;
			arrtemp[j][1] = arrtemp[NCols / 2 + j][1] * val;
			arrtemp[NCols / 2 + j][0] = temp[0] * val;
			arrtemp[NCols / 2 + j][1] = temp[1] * val;
		}
		arrtemp += NCols;
	}
}
//-----------------------------------------------------------------------------------------------------
void CChunk_cpu::fnc_roll_ffted(fftwf_complex* pcmparr_ffted, const int NRows, const int NCols)
{	
	fftwf_complex* arrtemp = pcmparr_ffted;
	for (int i = 0; i < NRows; ++i)
	{
		for (int j = 0; j < NCols / 2; ++j)
		{
			fftwf_complex temp;
			temp[0] = arrtemp[j][0];
			temp[1] = arrtemp[j][1];
			arrtemp[j][0] = arrtemp[NCols / 2 + j][0] ;
			arrtemp[j][1] = arrtemp[NCols / 2 + j][1] ;
			arrtemp[NCols / 2 + j][0] = temp[0] ;
			arrtemp[NCols / 2 + j][1] = temp[1] ;
		}
		arrtemp += NCols;
	}
}
//---------------------------------------------------------------------------------------
void CChunk_cpu::fnc_element_wise_mult(std::complex<float>* arr0, fftwf_complex* arr1, fftwf_complex* arrOut, const int LEn)
{
	for (int i = 0; i < LEn; ++i)
	{
		arrOut[ i ][0] = arr0[ i ].real() * arr1[  i ][0] - arr0[ i ].imag() * arr1[ i ][1];
		arrOut[ i ][1] = arr0[ i ].real() * arr1[ i ][1] + arr0[ i ].imag() * arr1[ i ][0];
	}
 }
//------------------------------------------------------------------------------------------

void  CChunk_cpu::transpose_unpadd(fftwf_complex* arin,fftwf_complex* fbuf)
{
	//fbuf = np.zeros((msamp, nsub, nchan), dtype = np.complex64)
	//	nfft, nsub, nchan, mbin = cp.shape
	int noverlap_per_channel = get_noverlap_per_channel();
	int mbin_adjusted = get_mbin_adjusted();
	const int nsub = m_nchan;
	const int nchan = m_len_sft;
	const int mbin = get_mbin();
	for (int ibin =0; ibin < mbin_adjusted; ++ibin)
		for (int ichan = 0; ichan < nchan; ++ichan)
		{
			for (int ifft = 0; ifft <m_nfft; ++ifft)
			{
				int num = 0;
				for (int isub = 0; isub < nsub; ++isub)
				{
					// Select bins from valid region and reverse the frequency axis
					int isamp = ibin + mbin_adjusted * ifft;
					int ibin_adjusted = ibin + noverlap_per_channel;
					fbuf[isamp * nsub * nchan + isub * nchan + nchan - ichan - 1][0] =
						arin[ifft * nsub * nchan * mbin + (nsub - isub - 1) * nchan * mbin + ichan * mbin + ibin_adjusted][0];
					fbuf[isamp * nsub * nchan + isub * nchan + nchan - ichan - 1][1] =
						arin[ifft * nsub * nchan * mbin + (nsub - isub - 1) * nchan * mbin + ichan * mbin + ibin_adjusted][1];
					++num;
						
				}
			}
		}
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
//----------------------------------------------











