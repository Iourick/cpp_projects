//#include "FdmtCpuF_omp.h"
//#include "FdmtCpuT_omp.h"
#include "HybridC_v0.h"

#include "StreamParams.h"
//#include "utilites.h"

#include "Constants.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <array>
#include <vector>
#include "npy.hpp"
#include "yr_cart.h"
#include <complex>
#include "CFdmtC.h"



#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif




using namespace std;
char str_wis_forward[10000] = { 0 };
char str_wis_backward[10000] = { 0 };
char str_wis_short[10000] = { 0 };
fftwf_plan create_wis(const unsigned int size,char*str, const bool bforward)
{
	fftwf_complex* in = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * size);
	fftwf_complex* out = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * size);
	fftwf_plan plan = NULL;
	if (bforward)
	{
		int size0 = size;
		plan = fftwf_plan_many_dft(1, &size0, 1,
			in, &size0,
			1, size,
			out, &size0,
			1, size,
			FFTW_FORWARD, FFTW_MEASURE);
		
		//plan = fftwf_plan_dft_1d(size, in, out, FFTW_FORWARD, FFTW_MEASURE);
	}
	else
	{
		plan = fftwf_plan_dft_1d(size, in, out, FFTW_BACKWARD, FFTW_MEASURE);
	}
	strcpy(str, fftwf_export_wisdom_to_string());	
	fftwf_free(in);
	fftwf_free(out);
	return plan;
}
//------------------------------------------------------------------

int fncHybridScan(float* parrSucessImagesBuff, int* piarrNumSuccessfulChunks, float *parrCoherent_d, int& quantOfSuccessfulChunks, CStreamParams* pStreamPars)
{
	fftwf_cleanup();
	const int NumChunks = ((pStreamPars->m_numEnd - pStreamPars->m_numBegin) + pStreamPars->m_lenChunk - 1) / pStreamPars->m_lenChunk;
	fftwf_complex* pRawSignalCur = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * pStreamPars->m_lenChunk);
	quantOfSuccessfulChunks = 0;
	// create plans for fft
	
	create_wis(pStreamPars->m_lenChunk, str_wis_forward, true);
	create_wis(pStreamPars->m_lenChunk, str_wis_backward, false);
	create_wis(pStreamPars->m_n_p, str_wis_short, true);
	
	size_t sz = sizeof(fftwf_complex);
	size_t sz1 = sizeof(complex<float>);
	// remains not readed elements
	int iremains = pStreamPars->m_lenarr;
	float val_coherent_d;
	for (int i = 0; i < NumChunks; ++i)
	{
		int length = (iremains < pStreamPars->m_lenChunk) ? iremains : pStreamPars->m_lenChunk;	

		if (length < pStreamPars->m_lenChunk)
		{
			create_wis(length, str_wis_forward, true);
			create_wis(length, str_wis_backward, false);
		}
		

		fread(pRawSignalCur, sizeof(fftwf_complex), length, pStreamPars->m_stream);
		float* poutImage = nullptr;
		if (nullptr != parrSucessImagesBuff)
		{
			poutImage = &parrSucessImagesBuff[pStreamPars->m_n_p * (pStreamPars->m_lenChunk / pStreamPars->m_n_p) * quantOfSuccessfulChunks];
		}
		if (fncSearchForHybridDedispersion(poutImage, pRawSignalCur, length, pStreamPars->m_n_p
			, pStreamPars->m_D_max, pStreamPars->m_f_min, pStreamPars->m_f_max, pStreamPars->m_SigmaBound
			, val_coherent_d))
			
		{
			piarrNumSuccessfulChunks[quantOfSuccessfulChunks] = i;
			parrCoherent_d[quantOfSuccessfulChunks] = val_coherent_d;
			++quantOfSuccessfulChunks;
		}
		++(pStreamPars->m_numCurChunk);
		iremains -= pStreamPars->m_lenChunk;
		
	}	
	// !	
	
	fftwf_free(pRawSignalCur);
	return 0;
}
//-------------------------------------------------------------
bool createOutImageForFixedNumberChunk(float* parr_fdmt_out, int* pargmaxRow, int* pargmaxCol, float* pvalSNR
	,float** pparrOutSubImage, int *piQuantRowsPartImage,CStreamParams* pStreamPars	, const int numChunk
	, const float VAlCoherent_d)
{
	int iremains = pStreamPars->m_lenarr - numChunk * pStreamPars->m_lenChunk;
	int lengthChunk = (iremains < pStreamPars->m_lenChunk) ? iremains : pStreamPars->m_lenChunk;
	int icols = lengthChunk / pStreamPars->m_n_p;
	
	fftwf_complex* pRawSignalCur = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * lengthChunk);

	fseek(pStreamPars->m_stream, numChunk * pStreamPars->m_lenChunk* sizeof(fftwf_complex), SEEK_CUR);
	fread(pRawSignalCur, sizeof(fftwf_complex), lengthChunk, pStreamPars->m_stream);

	bool bres = false;
	
	float valSigmaBound = pStreamPars->m_SigmaBound;
	
	fftwf_complex* pffted_rowsignal = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * lengthChunk);	

	// 1. create FFT
	fftwf_cleanup();	
	// Create the FFT plan
	fftw_import_wisdom_from_string(str_wis_forward);
	fftwf_plan plan = fftwf_plan_dft_1d(lengthChunk, pRawSignalCur, pffted_rowsignal, FFTW_FORWARD, FFTW_ESTIMATE);

	// Execute the FFT
	fftwf_execute(plan);
	fftwf_destroy_plan(plan);
	// !1

	// 2. create fdmt ones
	float* parr_fdmt_ones = (float*)malloc(lengthChunk * sizeof(float));
	for (int i = 0; i < lengthChunk; ++i)
	{
	parr_fdmt_ones[i] = 1.;
	}
	int IMaxDT = pStreamPars->m_IMaxDT;
	int N_p = pStreamPars->m_n_p;
	const float VAlFmin = pStreamPars->m_f_min;
	const float VAlFmax = pStreamPars->m_f_max;
	const float VAlD_max = pStreamPars->m_D_max;
	float* parrImNormalize = (float*)malloc(N_p * (lengthChunk / N_p) * sizeof(float));

	CFdmtC Fdmt(
		VAlFmin
		, VAlFmax
		, N_p		
		, lengthChunk / N_p
		, IMaxDT // quantity of rows of output image
	);

	Fdmt.process_image(parr_fdmt_ones, parrImNormalize, true);

	/*fncFdmt_cpuT_v0(parr_fdmt_ones, N_p, lengthChunk / N_p
	, VAlFmin, VAlFmax, IMaxDT, parrImNormalize);*/
	free(parr_fdmt_ones);
	// !2

	// 3.		
	float valConversionConst = DISPERSION_CONSTANT * (1. / (VAlFmin * VAlFmin) - 1. / (VAlFmax * VAlFmax)) * (VAlFmax - VAlFmin);
	float valN_d = VAlD_max * valConversionConst;
	// !3

	//4.
	fftwf_complex* pcarrCD_Out = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * lengthChunk);
	fftwf_complex* pcarrTemp = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * (lengthChunk / N_p) * N_p);
	float* parr_fdmt_inp = (float*)malloc(sizeof(float) * (lengthChunk / N_p) * N_p);
		
	createOutputFDMT(parr_fdmt_out, pffted_rowsignal, pcarrCD_Out, pcarrTemp, lengthChunk
	, N_p, parr_fdmt_inp, IMaxDT, DISPERSION_CONSTANT * VAlCoherent_d, VAlD_max, VAlFmin, VAlFmax);		

	float maxSig = -1.;
	int iargmax = -1;
	fncMaxSignalDetection(parr_fdmt_out, parrImNormalize, N_p, (lengthChunk / N_p)
	, pvalSNR, &iargmax);
	*pargmaxRow = iargmax / (lengthChunk / N_p);
	*pargmaxCol = iargmax % (lengthChunk / N_p);

	cutQuadraticSubImage(pparrOutSubImage, piQuantRowsPartImage, parr_fdmt_out, N_p, (lengthChunk / N_p), *pargmaxRow, *pargmaxCol);

	
	fftwf_free(pcarrTemp);
	fftwf_free(pcarrCD_Out);
	free(parr_fdmt_inp);
	
	free(parrImNormalize);
	return bres;

}
//-----------------------------------------------------------------------------------------
void cutQuadraticSubImage(float** pparrOutImage, int *piQuantRowsOutImage,float* InpImage, const int QInpImageRows, const int QInpImageCols
	, const int NUmCentralElemRow, const int NUmCentralElemCol)
{
	*piQuantRowsOutImage = (QInpImageRows < QInpImageCols) ? QInpImageRows : QInpImageCols;
	(*pparrOutImage) = (float*)realloc((*pparrOutImage), (*piQuantRowsOutImage) * (*piQuantRowsOutImage) * sizeof(float));
	float* p = (*pparrOutImage);
	if (QInpImageRows < QInpImageCols)
	{
		int numPart = NUmCentralElemCol / QInpImageRows;
		int numColStart = numPart * QInpImageRows;
		for (int i = 0; i < QInpImageRows; ++i)
		{
			memcpy(&p[i * QInpImageRows], &InpImage[i * QInpImageCols + numColStart], QInpImageRows * sizeof(float));
		}
		return;
    }
	int numPart = NUmCentralElemRow / QInpImageCols;
	int numStart = numPart * QInpImageCols;
	memcpy(p, &InpImage[numStart], QInpImageCols * QInpImageCols * sizeof(float));
}

//--------------------------------------------------------------
//INPUT:
// pffted_rowsignal - complex array, ffted 1-dimentional row signal, done from current chunk,  length = LEnChunk
// pcarrCD_Out - memory allocated comlex buffer to save output of coherent dedispersion function, nmed as fncCoherentDedispersion,
//				1- dimentional complex array, length = 	LEnChunk
// pcarrTemp - memory allocated comlex buffer to save output of STFT function, named as fncSTFT. 2-dimentional complex array
//            with dimensions = N_p x (LEnChunk / N_p)
// LEnChunk - length of input ffted signal pffted_rowsignal
// N_p - 
// parr_fdmt_inp - memory allocated float buffer to save input for FDMT function, dimentions = N_p x (LEnChunk / N_p)
// IMaxDT - the maximal delay (in time bins) of the maximal dispersion. Appears in the paper as N_{\Delta}
//            A typical input is maxDT = N_f
// VAlLong_coherent_d - is DispersionConstant* d, where d - is the dispersion measure.units: pc * cm ^ -3
// VAlD_max - maximal dispersion to scan, in units of pc cm^-3
// VAlFmin - the minimum freq, given in Mhz
// VAlFmax - the maximum freq,
//
// OUTPUT:
// parr_fdmt_out - float 2-dimensional array,with dimensions =  IMaxDT x (LEnChunk / N_p)
//int createOutputFDMT(float* parr_fdmt_out, fftwf_complex* pffted_rowsignal, fftwf_complex* pcarrCD_Out
//	, fftwf_complex* pcarrTemp
//	, const unsigned int LEnChunk, const unsigned int N_p, float* parr_fdmt_inp, const unsigned int IMaxDT
//	, const long double VAl_practicalD, const float VAlD_max, const float VAlFmin, const float VAlFmax)
//{
//	fncCoherentDedispersion(pcarrCD_Out, pffted_rowsignal, LEnChunk, VAl_practicalD, VAlFmin, VAlFmax);
//
//	/*std::vector<std::complex<float>> data(LEnChunk, 0);
//	memcpy(data.data(), pcarrCD_Out, LEnChunk * sizeof(std::complex<float>));
//	std::array<long unsigned, 1> leshape127{ LEnChunk };
//	npy::SaveArrayAsNumpy("pcarrCD_Out_cpu.npy", false, leshape127.size(), leshape127.data(), data);*/
//
//	fncSTFT(pcarrTemp, pcarrCD_Out, LEnChunk, N_p);
//
//	float sum = 0.;
//	int len = (LEnChunk / N_p) * N_p;
//	float arrMean[128] = { 0. }, arrDisp[128] = { 0. };
//	for (int i = 0; i < N_p; ++i)
//	{
//		for (int j = 0; j < (LEnChunk / N_p); ++j)
//		{
//			int n = i * (LEnChunk / N_p) + j;
//			float temp = pcarrTemp[n][0] * pcarrTemp[n][0] + pcarrTemp[n][1] * pcarrTemp[n][1];
//			arrMean[i] += temp;
//			arrDisp[i] += temp * temp;
//			parr_fdmt_inp[n] = temp;
//		}
//		arrMean[i] = arrMean[i] / (LEnChunk / N_p);
//		
//	}
//
//	float mean = 0., disp = 0.;
//	for (int i = 0; i < N_p; ++i)
//	{
//		mean += arrMean[i];
//		disp += arrDisp[i];// +arrMean[i] * arrMean[i];
//	}
//	mean = mean / N_p;
//	disp = sqrt((disp/ LEnChunk - mean * mean));
//	//---------------
//	for (int j = 0; j < len; ++j)
//	{		
//		float temp = pcarrTemp[j][0] * pcarrTemp[j][0] + pcarrTemp[j][1] * pcarrTemp[j][1];
//		sum += temp;
//		parr_fdmt_inp[j] = temp;
//	}
//
//	float val_mean = sum / ((float)len);
//	float valStdDev = fnsStdDev(parr_fdmt_inp, val_mean, len);
//
//	for (int j = 0; j < len; ++j)
//	{
//		parr_fdmt_inp[j] = (parr_fdmt_inp[j] - val_mean) / (0.25 * valStdDev);
//	}
//
//	
//
//	fncFdmt_cpuT_v0(parr_fdmt_inp, N_p, LEnChunk / N_p
//		, VAlFmin, VAlFmax, IMaxDT, parr_fdmt_out);
//
//	return 0;
//}
int createOutputFDMT(float* parr_fdmt_out, fftwf_complex* pffted_rowsignal, fftwf_complex* pcarrCD_Out
	, fftwf_complex* pcarrTemp
	, const unsigned int LEnChunk, const unsigned int N_p, float* parr_fdmt_inp, const unsigned int IMaxDT
	, const long double VAl_practicalD, const float VAlD_max, const float VAlFmin, const float VAlFmax)
{
	fncCoherentDedispersion(pcarrCD_Out, pffted_rowsignal, LEnChunk, VAl_practicalD, VAlFmin, VAlFmax);

	/*std::vector<std::complex<float>> data(LEnChunk, 0);
	memcpy(data.data(), pcarrCD_Out, LEnChunk * sizeof(std::complex<float>));
	std::array<long unsigned, 1> leshape127{ LEnChunk };
	npy::SaveArrayAsNumpy("pcarrCD_Out_cpu.npy", false, leshape127.size(), leshape127.data(), data);*/



	fncSTFT(pcarrTemp, pcarrCD_Out, LEnChunk, N_p);

	//memcpy(data.data(), pcarrTemp, LEnChunk * sizeof(std::complex<float>));
	////std::array<long unsigned, 1> leshape127{ LEnChunk };
	//npy::SaveArrayAsNumpy("pcarrTemp_cpu.npy", false, leshape127.size(), leshape127.data(), data);

	float sum = 0.;
	int len = (LEnChunk / N_p) * N_p;
	for (int j = 0; j < len; ++j)
	{
		float xx = pcarrTemp[j][0];
		//float temp = pcarrTemp[j].real() * pcarrTemp[j].real() + pcarrTemp[j].imag() * pcarrTemp[j].imag();
		float temp = pcarrTemp[j][0] * pcarrTemp[j][0] + pcarrTemp[j][1] * pcarrTemp[j][1];
		sum += temp;
		parr_fdmt_inp[j] = temp;
	}

	float val_mean = sum / ((float)len);
	float valStdDev = fnsStdDev(parr_fdmt_inp, val_mean, len);

	for (int j = 0; j < len; ++j)
	{
		parr_fdmt_inp[j] = (parr_fdmt_inp[j] - val_mean) / (0.25 * valStdDev);
	}

	//std::vector<float> data1(LEnChunk, 0);
	//memcpy(data1.data(), parr_fdmt_inp, LEnChunk * sizeof(float));
	////std::array<long unsigned, 1> leshape127{ LEnChunk };
	//npy::SaveArrayAsNumpy("parr_fdmt_inp.npy", false, leshape127.size(), leshape127.data(), data1);

	/*fncFdmt_cpuT_v0(parr_fdmt_inp, N_p, LEnChunk / N_p
		, VAlFmin, VAlFmax, IMaxDT, parr_fdmt_out);*/
	CFdmtC Fdmt(
		VAlFmin
		, VAlFmax
		, N_p		
		, LEnChunk / N_p
		, IMaxDT // quantity of rows of output image
	);

	Fdmt.process_image(parr_fdmt_inp, parr_fdmt_out, false);
	return 0;
}
//---------------------------------------------------------------------------------------------------------
void fncMaxSignalDetection(float* parr_fdmt_out, float* parrImNormalize, const unsigned int qRows, const unsigned int qCols
	, float* pmaxElement, int* argmax)
{
	const unsigned int len = qRows * qCols;		
	float* p = parr_fdmt_out;
	float* pn = parrImNormalize;
	for (int i = 0; i < len; ++i)
	{
		*p = (*p) / sqrt(((*pn) * 16. + 0.000001));
		++p;
		++pn;
	}
	
	(*pmaxElement) = (*max_element(parr_fdmt_out, parr_fdmt_out + len));
	*argmax = max_element(parr_fdmt_out, parr_fdmt_out + len) - parr_fdmt_out;
}

//-----------------------------------------------------------
void fncSTFT(fftwf_complex* pcarrOut, fftwf_complex* pRawSignalCur,  const unsigned int LEnChunk, int block_size)
{
	int qRows = LEnChunk / block_size;
	// allocate memory for temporary matrix
	fftwf_complex* pcarrS0 = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * qRows * block_size);

	fftwf_cleanup();
	
	fftw_import_wisdom_from_string(str_wis_short);
	fftwf_plan plan = fftwf_plan_dft_1d(block_size, pRawSignalCur, pcarrS0, FFTW_FORWARD, FFTW_ESTIMATE);
	fftwf_execute(plan);	
	
	for (int i = 1; i < qRows; ++i)
	{
		fftwf_execute_dft(plan, &pRawSignalCur[i * block_size], &pcarrS0[i * block_size]);
		
	}
	
	fftwf_destroy_plan(plan);
	fncMtrxTranspose_(pcarrOut, pcarrS0, qRows, block_size);
	
	fftwf_free(pcarrS0);
}

//---------------------------------------------------------------------------------
void fncCoherentDedispersion(fftwf_complex* pcarrCD_Out, fftwf_complex* pcarrffted_rowsignal
	, const unsigned int LEnChunk, const long double VAl_practicalD, const float VAlFmin, const float VAlFmax)
{	
	long double step = ((long double)VAlFmax - (long double)VAlFmin) / ((long double)LEnChunk);	

	fftwf_complex*  pcarrTemp = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * LEnChunk);
	fftwf_complex* pcarr = pcarrTemp;
	fftwf_complex* pcarr_row = pcarrffted_rowsignal;
	long double delfi = 0.;
	for (int i = 0; i < LEnChunk; ++i)
	{
		long double val_fi = (long double)VAlFmin + delfi;
		long double t3 = (VAl_practicalD / val_fi
			+ VAl_practicalD / ((long double)VAlFmax * (long double)VAlFmax) * ((long double)i) * step) * 2. * M_PI;		
		float val_x = cosl(t3);
		float val_y = -sinl(t3);
		(*pcarr)[0] = (float)(val_x * (*pcarr_row)[0] - val_y * (*pcarr_row)[1]); // Real part
		(*pcarr)[1] = (float)(val_x * (*pcarr_row)[1] + val_y * (*pcarr_row)[0]); // Imaginary part
		delfi += step;
		++pcarr;
		++pcarr_row;
	}
	

	fftwf_cleanup();

	fftw_import_wisdom_from_string(str_wis_backward);

	fftwf_plan plan = fftwf_plan_dft_1d(LEnChunk, pcarrTemp, pcarrCD_Out, FFTW_BACKWARD, FFTW_ESTIMATE);

	fftwf_execute(plan);
	fftwf_destroy_plan(plan);
	pcarr = pcarrCD_Out;
	for (int i = 0; i < LEnChunk; ++i)
	{
		(*pcarr)[0] = (*pcarr)[0] / (float)LEnChunk;
		(*pcarr)[1] = (*pcarr)[1] / (float)LEnChunk;
		++pcarr;
	}
	fftwf_free(pcarrTemp);
	pcarr = nullptr;
	pcarr_row = nullptr;

}
//-----------------------------------------------------------------------------------------------------------------
bool fncSearchForHybridDedispersion(float* poutImage, fftwf_complex* pRawSignalCur
	, const unsigned int LEnChunk, const unsigned int N_p
	, const float VAlD_max, const float VAlFmin, const float VAlFmax, float& valSigmaBound_, float& coherent_d)
{
	bool bres = false;
	coherent_d = -1.;
	float valSigmaBound = valSigmaBound_;	
	// 1. create FFT
	//fftwf_cleanup();
	fftwf_complex* pffted_rowsignal = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * LEnChunk);
	fftw_import_wisdom_from_string(str_wis_forward);
	
	fftwf_plan plan = fftwf_plan_dft_1d(LEnChunk, pRawSignalCur, pffted_rowsignal, FFTW_FORWARD, FFTW_ESTIMATE);
	fftwf_execute(plan);
	fftwf_destroy_plan(plan);	
	// !1


	// 2. create fdmt ones
	float* parr_fdmt_ones = (float*)malloc(LEnChunk * sizeof(float));
	for (int i = 0; i < LEnChunk; ++i)
	{
		parr_fdmt_ones[i] = 1.;
	}
	int IMaxDT = N_p;
	float* parrImNormalize = (float*)malloc(N_p * (LEnChunk / N_p) * sizeof(float));

	/*fncFdmt_cpuT_v0(parr_fdmt_ones, N_p, LEnChunk / N_p
		, VAlFmin, VAlFmax, IMaxDT, parrImNormalize);*/
	CFdmtC Fdmt(
		VAlFmin
		, VAlFmax
		, N_p		
		, LEnChunk / N_p
		, IMaxDT // quantity of rows of output image
	);

	Fdmt.process_image(parr_fdmt_ones, parrImNormalize, true);

	free(parr_fdmt_ones);
	// !2

	

	// 3.		
	float valConversionConst = DISPERSION_CONSTANT * (1. / (VAlFmin * VAlFmin) - 1. / (VAlFmax * VAlFmax)) * (VAlFmax - VAlFmin);
	float valN_d = VAlD_max * valConversionConst;
	int n_coherent = int(ceil(valN_d / (N_p * N_p)));
	cout << "n_coherent = " << n_coherent << endl;

	// !3

	//4.
	fftwf_complex* pcarrCD_Out = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * LEnChunk);
	fftwf_complex* pcarrTemp = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * (LEnChunk / N_p) * N_p);
	
	float* parr_fdmt_inp = (float*)malloc(sizeof(float) * (LEnChunk / N_p) * N_p);
	float* parr_fdmt_out = (float*)malloc(sizeof(float) * (LEnChunk / N_p) * N_p);
	
	
	//for (int iouter_d = 31; iouter_d < 32; ++iouter_d)
	for (int iouter_d = 0; iouter_d < n_coherent; ++iouter_d)
		{
		cout << "coherent iteration " << iouter_d << endl;

		long double valcur_coherent_d = ((long double)iouter_d) * ((long double)VAlD_max / ((long double)n_coherent));
		cout << "cur_coherent_d = " << valcur_coherent_d << endl;
		if (2 == iouter_d)
		{
			int uu = 0;
		}
		createOutputFDMT(parr_fdmt_out, pffted_rowsignal, pcarrCD_Out, pcarrTemp
			, LEnChunk, N_p, parr_fdmt_inp, IMaxDT
			, DISPERSION_CONSTANT * valcur_coherent_d, VAlD_max, VAlFmin, VAlFmax);		

		int len = (LEnChunk / N_p) * N_p;

		float maxSig= -1.;
		int iargmax =  -1;
		fncMaxSignalDetection(parr_fdmt_out, parrImNormalize, N_p, (LEnChunk / N_p)
			, &maxSig,  &iargmax);

		if (maxSig > valSigmaBound)
		{
			valSigmaBound = maxSig;
			coherent_d = valcur_coherent_d;
			cout << "!!!!!!! achieved score with " << valSigmaBound << "!!!!!!!" << endl;
			bres = true;
			if (nullptr != poutImage)
			{
				memcpy(poutImage, parr_fdmt_out, len * sizeof(float));
			}
			std::cout << "SNR = " << maxSig << endl;
			std::cout << "NUM ARGMAX = " << iargmax << endl;
			std::cout << "ROW ARGMAX = " << (int)(iargmax/(LEnChunk / N_p)) << endl;
			std::cout << "COLUMN ARGMAX = " << (int)(iargmax %(LEnChunk / N_p)) << endl;

		}
	}
	fftwf_free(pcarrTemp);
	fftwf_free(pcarrCD_Out);
	fftwf_free(pffted_rowsignal);
	free(parr_fdmt_inp);
	free(parr_fdmt_out);
	free(parrImNormalize);
	return bres;
}

	//----------------------------------------------------------------
	
	void fncMtrxTranspose(fdmt_type_* pArrout, fdmt_type_* pArrinp, const int QRowsInp, const int QColsInp)
	{
		fdmt_type_* pint = pArrinp;
		fdmt_type_* pout = pArrout;
		for (int i = 0; i < QRowsInp; ++i)
		{
			pout = pArrout + i;
			for (int j = 0; j < QColsInp; ++j)
			{
				*pout = *pint;
				++pint;
				pout += QRowsInp;
			}

		}
	}

	//----------------------------------------------------------------
	
	void fncMtrxTranspose_(fftwf_complex* pArrout, fftwf_complex* pArrinp, const int QRowsInp, const int QColsInp)
	{
		fftwf_complex* pint = pArrinp;
		fftwf_complex* pout = pArrout;
		for (int i = 0; i < QRowsInp; ++i)
		{
			pout = pArrout + i;
			for (int j = 0; j < QColsInp; ++j)
			{
				(*pout)[0] = (*pint)[0];
				(*pout)[1] = (*pint)[1];
				++pint;
				pout += QRowsInp;
			}

		}
	}
//-------------------------------------------------
void fncElementWiseModSq(float* parrOut, fftwf_complex* pcarrInp, unsigned int len)
{
	for (int i = 0; i < len; ++i)
	{
		parrOut[i] = pcarrInp[i][0] * pcarrInp[i][0] + pcarrInp[i][1] * pcarrInp[i][1];
	}
}
//-------------------------------------------------------------------

float fnsStdDev(fdmt_type_* parr_fdmt_inp, const float mean, unsigned int len)
{
	float sum = 0.;
	for (int i = 0; i < len; ++i)
	{
		sum += ((float)parr_fdmt_inp[i] - mean) * ((float)parr_fdmt_inp[i] - mean);
	}
	return sqrt(sum/ ((float)len));
}
//------------------------------------------------------------------

void fncFdmt_cpuT_v1(fdmt_type_* piarrImg, const int iImgrows
	, const int iImgcols, const float f_min
	, const  float f_max, const int imaxDT, fdmt_type_* piarrOut)
{
	
}
//-------------------------------------------------------------

void fncDisp(fdmt_type_* parr_fdmt_inp, unsigned int len, fdmt_type_& val_mean, fdmt_type_& val_V)
{
	val_mean = 0;
	fdmt_type_* p = parr_fdmt_inp;
	for (int i = 0; i < len; ++i)
	{
		val_mean += *p;
		++p;
	}

	val_mean = val_mean / len;

	val_V = 0;
	p = parr_fdmt_inp;
	for (int i = 0; i < len; ++i)
	{
		fdmt_type_ temp = *p - val_mean;
		val_V += temp * temp;
		++p;
	}
	val_V /= ((fdmt_type_)len);

}