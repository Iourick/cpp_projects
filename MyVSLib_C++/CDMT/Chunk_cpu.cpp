#include "Chunk_cpu.h"
#include <chrono>
#include <cmath>
//#include "npy.hpp"
#include "FdmtCpu.h"
#include <algorithm>

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
	// 1.
	const int ndm = m_coh_dm_Vector.size();

	//2.create dc vector
	m_dc_Vector.resize(ndm * m_nchan * m_nbin);

	//auto start = std::chrono::high_resolution_clock::now();
	compute_chirp_channel(&m_dc_Vector, &m_coh_dm_Vector);
	//auto end = std::chrono::high_resolution_clock::now();
	//auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	//std::cout << "Time taken by function compute_chirp_channel: " << duration.count()/* / ((double)num) */<< " milliseconds" << std::endl;
	
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
bool CChunk_cpu::process(void* pcmparrRawSignalCur
	, std::vector<COutChunkHeader>* pvctSuccessHeaders, std::vector<std::vector<float>>* pvecImg)
{	
	// 1. 
	const int mbin = get_mbin();
	const int noverlap_per_channel = get_noverlap_per_channel();
	const int mbin_adjusted = get_mbin_adjusted();
	const int msamp = get_msamp();
	const int mchan = m_nchan * m_len_sft;
	// 1!
	
	// 2. Forward FFT execution
	fftwf_cleanup();
	fftw_import_wisdom_from_string(m_str_wis_forw);
	int  howmany = m_nchan * m_npol / 2 * m_nfft;
	fftwf_plan plan  = fftwf_plan_many_dft(1, &m_nbin, howmany,
		(fftwf_complex*)pcmparrRawSignalCur, &m_nbin, 1, m_nbin, (fftwf_complex*)pcmparrRawSignalCur, &m_nbin, 1, m_nbin, FFTW_FORWARD, FFTW_ESTIMATE);
	fftwf_execute(plan);
	fftwf_destroy_plan(plan);
	//2!	
	
	//3. roll and normalize ffted signals
	fnc_roll_and_normalize_ffted((fftwf_complex*)pcmparrRawSignalCur, m_nchan * m_npol / 2 * m_nfft, m_nbin);
	//3!

	// 4. memory allocation for buffers
	fftwf_complex*  parrElemWiseMulted = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * m_nchan * m_npol / 2 * m_nfft * m_nbin);
	
	fftwf_complex* fbuf = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * msamp* m_nchan * m_len_sft  * m_npol / 2 );
	float* parr_wfall = (float*)malloc(sizeof(float) * msamp * m_nchan * m_len_sft);
	
	fftwf_complex* parr_ffted = (fftwf_complex*)pcmparrRawSignalCur;
	float* parr_fdmt_res = (float*)malloc(m_len_sft  * msamp * sizeof(float));


	float* parr_cdmt_transform = nullptr;

	std::vector<std::vector<float>>* pvecImg_temp = nullptr;
	if (nullptr != pvecImg)
	{
		pvecImg->resize(m_coh_dm_Vector.size() );
		for (auto& row : *pvecImg)
		{
			row.resize(msamp * m_len_sft);
		}		
		pvecImg_temp = pvecImg;
	}
	else 
	{
		pvecImg_temp = new std::vector<std::vector<float>>;
		pvecImg_temp->resize(m_coh_dm_Vector.size() );
		for (auto& row : *pvecImg_temp)
		{
			row.resize(msamp * m_len_sft);
		}
	}
	
	// !4

	// 5. define FDMT instance
	CFdmtCpu  fdmt(
		m_Fmin
		, m_Fmax
		,  m_len_sft * m_nchan// quant channels/rows of input image, including consisting of zeroes
		, msamp
		, m_len_sft// quantity of rows of output image
	);
	//fdmt.process_image(nullptr, parr_fdmt_norm, true);
	// 5!
	
	// 6. main loop
	for (int idm = 0; idm < m_coh_dm_Vector.size(); ++idm)
	{
		// 7. elementwise multiplication
		std::complex<float>* pnt = m_dc_Vector.data();
#pragma omp parallel //
		{
			pnt += m_nchan * m_nbin * idm;
			for (int i = 0; i < m_npol / 2; ++i)
			{
				for (int j = 0; j < m_nfft; ++j)
				{
					fnc_element_wise_mult(pnt, parr_ffted + i * m_nfft * m_nchan * m_nbin + j * m_nchan * m_nbin, parrElemWiseMulted + i * m_nfft * m_nchan * m_nbin + j * m_nchan * m_nbin, m_nchan * m_nbin);
				}
			}
		}
		// 7!

		//8. roll and normalize ffted signals
		fnc_roll_ffted(parrElemWiseMulted, m_nchan * m_npol / 2 * m_len_sft* m_nfft, m_nbin/ m_len_sft);
		//8!		
		
		// 9. SFFT
		fftwf_cleanup();
		fftw_import_wisdom_from_string(m_str_wis_back);
		int n = m_nbin / m_len_sft, howmany = m_nchan * m_npol / 2 * m_nfft * m_len_sft;
		fftwf_plan plan = fftwf_plan_many_dft(1, &n, howmany,
			parrElemWiseMulted, &n, 1, n, parrElemWiseMulted, &n, 1, n, FFTW_BACKWARD, FFTW_ESTIMATE);
		fftwf_execute(plan);
		fftwf_destroy_plan(plan);
		// 9!
		
		// 10. normalization of parrElemWiseMulted,
		float coef = 1.0f / ((float)n);
#pragma omp parallel //
		{
			for (int i = 0; i < m_nchan * m_nfft * m_nbin * m_npol / 2; ++i)
			{
				parrElemWiseMulted[i][0] *= coef;
				parrElemWiseMulted[i][1] *= coef;
			}
		}
		// 10!	
		
		// 11.	unpadding and transposition
		transpose_unpadd(parrElemWiseMulted,  fbuf);
		transpose_unpadd(parrElemWiseMulted + m_nchan * m_nfft * m_nbin,  fbuf + msamp * m_nchan * m_len_sft);
		// 11!
		
		//12. calc intensity matrix already transposed
		
		memset(parr_wfall, 0, sizeof(float) * msamp * m_nchan * m_len_sft);
		const int NRows = m_nchan * m_len_sft;
		const int NCols = msamp;
		const int NEl = NRows * NCols;
#pragma omp parallel
	{
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
	}
		// 12!		

	// 13. dedisperse		
		fnc_dedisperse(parr_wfall, m_coh_dm_Vector[idm], m_tsamp * m_len_sft);	
	// 13!	
		
	// 14. FDMT-processing		
		//fdmt.process_image(parr_wfall, &parr_cdmt_transform[m_len_sft * msamp * idm], false);
		
		fdmt.process_image(parr_wfall, pvecImg_temp->at(idm).data(), false);
		
		//pvecImg->resize(m_coh_dm_Vector.size() * m_len_sft, msamp);
	// 14!			
	}	
			//std::vector<float> vec(parr_cdmt_transform, parr_cdmt_transform + msamp * m_len_sft * m_coh_dm_Vector.size());
			//// Accessing elements
			//std::array<long unsigned, 2> leshape127{ m_len_sft * m_coh_dm_Vector.size()  ,  msamp };
			//npy::SaveArrayAsNumpy("parr_wfall.npy", false, leshape127.size(), leshape127.data(), vec);
			
	fftwf_free(parrElemWiseMulted);
	fftwf_free(fbuf);
	free(parr_wfall);
	free(parr_fdmt_res);

	if (nullptr != pvecImg)
	{
		pvecImg_temp = nullptr;
	}
	else
	{
		free(pvecImg_temp);
	}
	return true;
}
//------------------------------------------------------------------- 
template <typename T>
void CChunk_cpu::roll_(T* arr, const int lenarr, const int ishift)
{
	if (ishift == 0)
	{
		return;
	}
	T* arr0 = (T*)malloc(ishift * sizeof(T));
	T* arr1 = (T*)malloc((lenarr - ishift) * sizeof(T));
	memcpy(arr0, arr + lenarr - ishift, ishift * sizeof(T));
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
#pragma omp parallel
	{
		for (int i = 0; i < nchans; ++i)
		{
			double temp = (double)(nchans - 1 - i) * foff + f_min;
			double temp1 = 4.148808e3 * dm * (1.0 / (f_min * f_min) - 1.0 / (temp * temp));
			int ishift = round(temp1 / tsamp);
			roll_(parr_wfall + i * msamp, msamp, ishift);
		}
	}	
}
//---------------------------------------------------------------------------
// INPUT:
// OUTPUT:
//parr_dc
void CChunk_cpu::compute_chirp_channel(std::vector<std::complex<float>>* parr_dc, const  std::vector<float>* parr_coh_dm)
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
	std::vector<double>arr_freqs_chan(m_nchan * m_len_sft);
	std::vector<double>arr_temp(m_len_sft);
#pragma omp parallel // 
	{
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
	}
	// 3!

	// 4. arr_bin_freqs && taper computation
	std::vector<double>arr_bin_freqs(mbin);
	std::vector<double>taper(mbin);
#pragma omp parallel // 
	{
		for (int i = 0; i < mbin; ++i)
		{
			arr_bin_freqs[i] = -0.5 * bw_chan + (i + 0.5) * bw_chan / mbin;
			taper[i] = 1.0 / sqrt(1.0 + pow(arr_bin_freqs[i] / (0.47 * bw_chan), 80));
		}
	}
	// 4!

	// element wise mult
	

	std::complex<double>cmpi(0.0, 1.0);
#pragma omp parallel // 
	{
		for (int idm = 0; idm < ndm; ++idm)
			for (int ichan = 0; ichan < m_nchan; ++ichan)
			{
				for (int isft = 0; isft < m_len_sft; ++isft)
				{
					double temp0 = arr_freqs_chan[ichan * m_len_sft + isft];
					int i0 = idm * m_nchan * m_len_sft * mbin + ichan * m_len_sft * mbin + +isft * mbin;				
					for (int ibin = 0; ibin < mbin; ++ibin)
					{
						double temp1 = arr_bin_freqs[ibin] / temp0;
						double phase_delay
							= (*parr_coh_dm)[idm] * temp1 * temp1 / (temp0 + arr_bin_freqs[ibin]) * 4.148808e9;
						(*parr_dc)[i0 + ibin]
							= std::exp(-2.0 * cmpi * M_PI * phase_delay) * taper[ibin];
					}				
				}
			}
		} // ! #pragma omp parallel


	/*std::array<long unsigned, 4> leshape127 {ndm, m_nchan,  m_len_sft,mbin };
	npy::SaveArrayAsNumpy("chirp_c.npy", false, leshape127.size(), leshape127.data(), *parr_dc);
	int ii = 0;*/
}
//-----------------------------------------------------------------------------------------------------
void CChunk_cpu::fnc_roll_and_normalize_ffted(fftwf_complex* pcmparr_ffted, const int NRows, const int NCols)
{
	float val = 1.0f / ((float)NCols);
	fftwf_complex* arrtemp = pcmparr_ffted;
#pragma omp parallel // 
	{
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
}
//-----------------------------------------------------------------------------------------------------
void CChunk_cpu::fnc_roll_ffted(fftwf_complex* pcmparr_ffted, const int NRows, const int NCols)
{	
	fftwf_complex* arrtemp = pcmparr_ffted;
#pragma omp parallel //
	{
		for (int i = 0; i < NRows; ++i)
		{
			for (int j = 0; j < NCols / 2; ++j)
			{
				fftwf_complex temp;
				temp[0] = arrtemp[j][0];
				temp[1] = arrtemp[j][1];
				arrtemp[j][0] = arrtemp[NCols / 2 + j][0];
				arrtemp[j][1] = arrtemp[NCols / 2 + j][1];
				arrtemp[NCols / 2 + j][0] = temp[0];
				arrtemp[NCols / 2 + j][1] = temp[1];
			}
			arrtemp += NCols;
		}
	}
}
//---------------------------------------------------------------------------------------
void CChunk_cpu::fnc_element_wise_mult(std::complex<float>* arr0, fftwf_complex* arr1, fftwf_complex* arrOut, const int LEn)
{
#pragma omp parallel
	{
		for (int i = 0; i < LEn; ++i)
		{
			arrOut[i][0] = arr0[i].real() * arr1[i][0] - arr0[i].imag() * arr1[i][1];
			arrOut[i][1] = arr0[i].real() * arr1[i][1] + arr0[i].imag() * arr1[i][0];
		}
	}
 }
//------------------------------------------------------------------------------------------

void  CChunk_cpu::transpose_unpadd(fftwf_complex* arin,fftwf_complex* fbuf)
{	
	int noverlap_per_channel = get_noverlap_per_channel();
	int mbin_adjusted = get_mbin_adjusted();
	const int nsub = m_nchan;
	const int nchan = m_len_sft;
	const int mbin = get_mbin();
#pragma omp parallel
	{
		for (int ibin = 0; ibin < mbin_adjusted; ++ibin)
		{
			int ibin_adjusted = ibin + noverlap_per_channel;
			for (int ichan = 0; ichan < nchan; ++ichan)
			{
				for (int ifft = 0; ifft < m_nfft; ++ifft)
				{
					int isamp = ibin + mbin_adjusted * ifft;
					int num = 0;
					for (int isub = 0; isub < nsub; ++isub)
					{
						// Select bins from valid region and reverse the frequency axis					
						fbuf[isamp * nsub * nchan + isub * nchan + nchan - ichan - 1][0] =
							arin[ifft * nsub * nchan * mbin + (nsub - isub - 1) * nchan * mbin + ichan * mbin + ibin_adjusted][0];
						fbuf[isamp * nsub * nchan + isub * nchan + nchan - ichan - 1][1] =
							arin[ifft * nsub * nchan * mbin + (nsub - isub - 1) * nchan * mbin + ichan * mbin + ibin_adjusted][1];
						++num;
					}
				}
			}
		}
	}
}

//
//
//void multi_windowing_cpu(fdmt_type_* arr, fdmt_type_* norm, const int  Cols
//	, const int WndWidth, float* pAuxArray, int* pAuxIntArray, int* pWidthArray, const dim3 gridSize, const dim3 blockSize)
//{
//	const int NUmGrigRows = gridSize.y;
//	const int NUmGrigCols = gridSize.x;
//	float sum2 = 0.;
//	float norm2 = 0.;
//	for (int i = 0; i < NUmGrigRows; ++i)
//	{
//
//		for (int j = 0; j < NUmGrigCols; ++j)
//		{
//			if ((j == (NUmGrigCols - 1)) && (i == 1))
//			{
//				int yyy = 0;
//			}
//			pAuxArray[i * NUmGrigCols + j] = 1. - FLT_MAX;
//			int num_begin = i * Cols + j * blockSize.x;
//			for (int k = 0; k < blockSize.x; ++k)
//			{
//
//				for (int iw = 1; iw <= WndWidth; ++iw)
//				{
//
//					sum2 = 0.;
//					norm2 = 0.;
//					for (int q = 0; q < iw; ++q)
//					{
//						if (j * blockSize.x + k + q >= Cols)
//						{
//							sum2 = 1. - FLT_MAX;
//							norm2 = 1.;
//							break;
//						}
//
//						float t = norm[num_begin + k + q] + 0.00001;// max((float)norm[num_begin + k + q], 1.);
//						sum2 += ((float)arr[num_begin + k + q]) / sqrt(t);
//
//					}
//					sum2 = sum2 / sqrt((float)iw);
//					if (sum2 > pAuxArray[i * NUmGrigCols + j])
//					{
//						pAuxArray[i * NUmGrigCols + j] = sum2;
//						pAuxIntArray[i * NUmGrigCols + j] = num_begin + k;
//						pWidthArray[i * NUmGrigCols + j] = iw;
//					}
//
//				}
//
//			}
//
//		}
//	}
//}
////----------------------------------------------
// bool CChunk_cpu::detect_signal(fdmt_type_* arr, fdmt_type_* norm, const int Rows, const int  Cols, structOutDetection *pstrOut)
//{	
//	float sum2 = 0.;
//	float norm2 = 0.;
//	std::vector< fdmt_type_ >vct_RowMax(Rows);
//	std::vector<  structOutDetection>vct_structArgMax(Rows);
//	
//	for (int i = 0; i < Rows; ++i)
//	{
//		detect_signal_in_row(&arr[i * Cols], &norm[i * Cols], Cols, &vct_RowMax[i], &vct_structArgMax[i]);
//		
//	}
//	auto maxElementIterator = std::max_element(vct_RowMax.begin(), vct_RowMax.end());
//
//	int iargMaxRow = std::distance(vct_RowMax.begin(), maxElementIterator);
//	
//	if ((pstrOut->snr) < m_sigma_bound)
//	{
//		return false;
//	}
//	*pstrOut = vct_structArgMax[iargMaxRow];
//	
//	return true;
//}
// //-------------------------------------------------------------------------------
// void CChunk_cpu::detect_signal_in_row(fdmt_type_* arr, fdmt_type_* norm, const int  Cols, const int  NumRow, structOutDetection* pstrOut)
// {
//	 std::vector<float> vct_Cum_ratio(Cols) ;
//	 vct_Cum_ratio[0] = (float)arr[0] / (1.0E-12 + sqrt(norm[0]));
//	 for (int i = 1; i < Cols; ++i)
//	 {		 
//		 vct_Cum_ratio[i] = vct_Cum_ratio[i - 1] + (float)arr[i] / (1.0E-12 + sqrt(norm[i]));
//	 }
//	 vct_Cum_ratio.insert(vct_Cum_ratio.begin(), 0.0f);
//	 vct_Cum_ratio.insert(vct_Cum_ratio.end(), m_length_sum_wnd, vct_Cum_ratio.back());
//	 
//
//	 pstrOut->irow = NumRow;
//	 pstrOut->icol = 0;
//	 pstrOut->iwidth = 1;
//	 pstrOut->snr = (float)arr[0] / ((float)norm[0] + 1.0E-10);
//
//	 for (int iwnd = 1; iwnd < (m_length_sum_wnd + 1); ++iwnd)
//	 {
//		 float  temp0 = sqrt((float)iwnd);
//		 for (int in = 0; in < Cols; ++in)
//		 {
//			 float temp = (vct_Cum_ratio[in + iwnd] - vct_Cum_ratio[in]) / temp0;
//			 if (temp > pstrOut->snr)
//			 {
//				 pstrOut->snr = temp;
//				 pstrOut->icol = in;
//				 pstrOut->iwidth = iwnd;
//			 }
//		 }
//		 
//	 }
// }












