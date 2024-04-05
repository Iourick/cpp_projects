#include "CFdmtC.h"
#include <math.h>
#include <array>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

CFdmtC::~CFdmtC()
{
	if (m_iNumIter > 0)
	{
		for (int i = 0; i < (m_iNumIter + 1); ++i)
		{
			if (m_pparrFreq[i])
			{
				free(m_pparrFreq[i]);
			}

			if (m_pparrRowsCumSum[i])
			{
				free(m_pparrRowsCumSum[i]);
			}			

		}
		free(m_pparrFreq);
		free(m_pparrRowsCumSum);
		if (m_parrQuantMtrx)
		{
			free(m_parrQuantMtrx);
		}
	}
}
//---------------------------------------
CFdmtC::CFdmtC()
{
	m_Fmin = 0;
	m_Fmax = 0;
	m_nchan = 0;	
	m_cols = 0;
	m_imaxDt = 0;
	m_pparrRowsCumSum = NULL;
	m_pparrFreq = NULL;
	m_parrQuantMtrx =  NULL ;
	m_iNumIter = 0;
}
//-----------------------------------------------------------

CFdmtC::CFdmtC(const  CFdmtC& R)
{
	m_Fmin = R.m_Fmin;
	m_Fmax = R.m_Fmax;
	m_nchan = R.m_nchan;	
	m_cols = R.m_cols;
	m_imaxDt = R.m_imaxDt;

	m_iNumIter = R.m_iNumIter;

	m_parrQuantMtrx = (int*)malloc((R.m_iNumIter + 1) * sizeof(int));
	memcpy(m_parrQuantMtrx, R.m_parrQuantMtrx, (R.m_iNumIter + 1) * sizeof(int));
	

	m_pparrFreq = (float**)malloc((R.m_iNumIter + 1) * sizeof(float*)); 
	for (int i = 0; i < (R.m_iNumIter + 1); ++i) {
		m_pparrFreq[i] = (float*)malloc(R.m_parrQuantMtrx[i] * sizeof(float));
		memcpy(m_pparrFreq[i], R.m_pparrFreq[i], R.m_parrQuantMtrx[i] * sizeof(float));
	}

	m_pparrRowsCumSum = (int**)malloc((R.m_iNumIter + 1) * sizeof(int*)); 
	for (int i = 0; i < (R.m_iNumIter + 1); ++i)
	{
		m_pparrRowsCumSum[i] = (int*)malloc(R.m_parrQuantMtrx[i] * sizeof(int));
		memcpy(m_pparrRowsCumSum[i], R.m_pparrRowsCumSum[i], R.m_parrQuantMtrx[i] * sizeof(int));
	}
}
//-------------------------------------------------------------------
CFdmtC& CFdmtC::operator=(const CFdmtC& R)
{
	if (this == &R)
	{
		return *this;
	}
	m_Fmin = R.m_Fmin;
	m_Fmax = R.m_Fmax;
	m_nchan = R.m_nchan;	
	m_cols = R.m_cols;
	m_imaxDt = R.m_imaxDt;
	m_iNumIter = R.m_iNumIter;

	
	for (int i = 0; i < m_iNumIter; ++i)
	{
		free(m_pparrFreq[i]);
		free(m_pparrRowsCumSum[i]);
	}
	free(m_pparrFreq);
	free(m_pparrRowsCumSum);
	

	m_pparrFreq = (float**)malloc((R.m_iNumIter + 1) * sizeof(float*));
	for (int i = 0; i < (R.m_iNumIter + 1); ++i) {
		m_pparrFreq[i] = (float*)malloc(R.m_parrQuantMtrx[i] * sizeof(float));
		memcpy(m_pparrFreq[i], R.m_pparrFreq[i], R.m_parrQuantMtrx[i] * sizeof(float));
	}

	m_pparrRowsCumSum = (int**)malloc((R.m_iNumIter + 1) * sizeof(int*));
	for (int i = 0; i < (R.m_iNumIter + 1); ++i)
	{
		m_pparrRowsCumSum[i] = (int*)malloc(R.m_parrQuantMtrx[i] * sizeof(int));
		memcpy(m_pparrRowsCumSum[i], R.m_pparrRowsCumSum[i], R.m_parrQuantMtrx[i] * sizeof(int));
	}

	free(m_parrQuantMtrx);
	m_parrQuantMtrx = (int*)malloc((R.m_iNumIter + 1) * sizeof(int));

	memcpy(m_parrQuantMtrx, R.m_parrQuantMtrx, (R.m_iNumIter + 1) * sizeof(int));
	return *this;
}

//--------------------------------------------------------------------
CFdmtC::CFdmtC(
	  const float Fmin
	, const float Fmax	
	, const int nchan // quant channels/rows of input image, including consisting of zeroes
	, const int cols		
    , const int imaxDt // quantity of rows of output image
)
{	
	m_nchan = nchan;
	m_Fmin = Fmin;
	m_Fmax = Fmax;
	m_cols = cols;
	m_imaxDt = imaxDt;

	create_config(m_pparrRowsCumSum,m_pparrFreq, &m_parrQuantMtrx,&m_iNumIter);
}



////-------------------------------------------------------------------------
void CFdmtC::process_image(fdmt_type_* piarrImgInp, fdmt_type_* piarrImgOut, const bool b_ones)
{	
	//1. declare pointers 
	fdmt_type_* p0 = 0;
	fdmt_type_* p1 = 0;
	fdmt_type_* piarrOut_0 = 0;
	fdmt_type_* piarrOut_1 = 0;
	// !1

	// 2. allocate memory 
	piarrOut_0 = (fdmt_type_*)calloc((m_pparrRowsCumSum[0])[m_parrQuantMtrx[0]] * m_cols, sizeof(fdmt_type_));
	piarrOut_1 = (fdmt_type_*)calloc((m_pparrRowsCumSum[1])[m_parrQuantMtrx[1]] * m_cols, sizeof(fdmt_type_));
	// !2

   // 3. call initialization func
	const int ideltaT = calc_deltaT(m_Fmin, m_Fmin + (m_Fmax - m_Fmin) / m_nchan);
	fnc_initC(piarrImgInp, ideltaT, piarrOut_0, b_ones);
	// !3

	// 4.pointers fixing
	p0 = piarrOut_0;
	p1 = piarrOut_1;
	// 4!

	// 5. calculations
	for (int iit = 1; iit < (1 +m_iNumIter); ++iit)
	{
		if (2 == iit)
		{
			int yy = 0;
		}
		fncFdmtIterationC(p0, iit, p1);
		// exchange order of pointers
		fdmt_type_* pt = p0;
		p0 = p1;
		p1 = pt;
	}
	// !5
	memcpy(piarrImgOut, p0, m_cols * m_imaxDt * sizeof(fdmt_type_));	
	free(piarrOut_0);
	free(piarrOut_1);
}
//------------------------------------------------------------------------------
void  CFdmtC::create_config(int** &pparrRowsCumSum, float** &pparrFreq, int** pparrQuantMtrx,int* piNumIter)
{
	// 1. calculation iterations quanttity *piNumIter and array *pparrQuantMtrx of quantity submatrices for each iteration
	// *pparrQuantMtrx  has length = *piNumIter +1
	// (*pparrQuantMtrx  has length)[0] = m_nchan , for initialization
	*piNumIter = calc_quant_iterations_and_lengthSubMtrxArray(pparrQuantMtrx);
	// 1!

	// 2. memory allocation for 2 auxillary arrays
	int* iarrQuantMtrx = *pparrQuantMtrx;

	pparrFreq = (float**)malloc((*piNumIter  +1) * sizeof(float*)); // Allocate memory for m pointers to int

	pparrRowsCumSum = (int**)malloc((*piNumIter +1) * sizeof(int*));
	
	for (int i = 0; i < (*piNumIter + 1); ++i)
	{
		pparrFreq[i] = (float*)malloc((iarrQuantMtrx[i] + 1) * sizeof(float));
		pparrRowsCumSum[i] = (int*)malloc((iarrQuantMtrx[i] + 1) * sizeof(int));
	}
	
	// 2!
	
	// 3. initialization 0 step	 
	float* arrFreq = pparrFreq[0];	
	//int* iarrRowsCumSum = pparrRowsCumSum[0];

	int* iarrQntSubmtrxRows = (int*)malloc(m_nchan * sizeof(int));

	int* iarrQntSubmtrxRowsCur = (int*)malloc(m_nchan * sizeof(int));

	const int ideltaT = calc_deltaT(m_Fmin, m_Fmin + (m_Fmax - m_Fmin) / m_nchan);
	for (int i = 0; i < m_nchan; ++i)
	{
		iarrQntSubmtrxRows[i] = ideltaT + 1;
		arrFreq[i] = m_Fmin + i * (m_Fmax - m_Fmin) / m_nchan;
	}
	arrFreq[m_nchan] = m_Fmax;
	calcCumSum_(iarrQntSubmtrxRows, iarrQuantMtrx[0], pparrRowsCumSum[0]);
	// 3!

	// 4. main loop. filling 2 config arrays	
	for (int i = 1; i < *piNumIter + 1; ++i)
	{
		
		calcNextStateConfig(iarrQuantMtrx[i - 1], iarrQntSubmtrxRows, pparrFreq[i - 1]
			, iarrQuantMtrx[i], iarrQntSubmtrxRowsCur, pparrFreq[i]);
		memcpy(iarrQntSubmtrxRows, iarrQntSubmtrxRowsCur, iarrQuantMtrx[i] * sizeof(int));
		calcCumSum_(iarrQntSubmtrxRowsCur, iarrQuantMtrx[i], pparrRowsCumSum[i]);
	}
	
	// 4!
	free(iarrQntSubmtrxRowsCur);
	free(iarrQntSubmtrxRows);	
}

//-----------------------------------------------------------------------------------------
void CFdmtC::fncFdmtIterationC(fdmt_type_* p0, const int  iit, fdmt_type_* p1)
{
	// 1. extract config for previous mtrix (p0 matrix)  
	int quantSubMtrx = m_parrQuantMtrx[iit - 1]; // quant of submatixes
	int* iarrCumSum = m_pparrRowsCumSum[iit - 1];
	float* arrFreq = m_pparrFreq[iit - 1];
	// 1!  

	// 2. extract config for curent matrix (p0 matrix)
	int quantSubMtrxCur = m_parrQuantMtrx[iit]; // quant of submatixes
	int* iarrCumSumCur = m_pparrRowsCumSum[iit];
	float* arrFreqCur = m_pparrFreq[iit];
	// 2! 

	// 3. combining 2  adjacent matrices
#pragma omp parallel // 
	{
		for (int i = 0; i < quantSubMtrxCur; ++i)
		{
			fdmt_type_* pout = &p1[iarrCumSumCur[i] * m_cols];
			fdmt_type_* pinp0 = &p0[iarrCumSum[i * 2] * m_cols];

			if ((i * 2 + 1) >= quantSubMtrx)
			{
				int quantLastSubmtrxRows = iarrCumSum[quantSubMtrx] - iarrCumSum[quantSubMtrx - 1];
				memcpy(pout, pinp0, quantLastSubmtrxRows * m_cols * sizeof(fdmt_type_));
				break;
			}

			fdmt_type_* pinp1 = &p0[iarrCumSum[i * 2 + 1] * m_cols];
			int quantSubtrxRowsCur = iarrCumSumCur[i + 1] - iarrCumSumCur[i];
			float coeff0 = fnc_delay(arrFreq[2 * i], arrFreq[2 * i + 1]) / fnc_delay(arrFreq[2 * i], arrFreq[2 * i + 2]);
			float coeff1 = fnc_delay(arrFreq[2 * i + 1], arrFreq[2 * i + 2]) / fnc_delay(arrFreq[2 * i], arrFreq[2 * i + 2]);
			for (int j = 0; j < quantSubtrxRowsCur; ++j)
			{
				int j0 = (int)(coeff0 * ((float)j));
				int j1 = (int)(coeff1 * ((float)j));

				for (int k = 0; k < m_cols; ++k)
				{					
					pout[j * m_cols + k] = pinp0[j0 * m_cols + k];
					if ((k - j0) >= 0)
					{
						pout[j * m_cols + k] += pinp1[j1 * m_cols + k - j0];
					}
				}
			}
		}
	}
	// 3!


	return;
}
//--------------------------------------------------------------------------------------

void  CFdmtC::fnc_initC(fdmt_type_* piarrImg, const int IDeltaT, fdmt_type_* piarrOut, bool b_ones )
{

	memset(piarrOut, 0, m_nchan * m_cols * (IDeltaT + 1) * sizeof(fdmt_type_ ));
	if (!b_ones)
	{
#pragma omp parallel // 
		{
			for (int i = 0; i < m_nchan; ++i)
			{
				{
					memcpy(&piarrOut[i * (IDeltaT + 1) * m_cols], &piarrImg[i * m_cols]
						, m_cols * sizeof(fdmt_type_));
				}
			}
		}
	}
	else
	{
		float temp = 1.;
		fdmt_type_ t = (fdmt_type_)temp;

#pragma omp parallel // 
		{
			for (int i = 0; i < m_nchan; ++i)
			{
				for (int j=0; j < m_cols;++j)
				{

					piarrOut[i * (IDeltaT + 1) * m_cols + j] = t;
					
				}
			}
		}
	}
#pragma omp parallel // OMP (
	{ 
		for (int i_dT = 1; i_dT < (IDeltaT + 1); ++i_dT)
			for (int iF = 0; iF < m_nchan; ++iF)
			{
				fdmt_type_* result = &piarrOut[iF * (IDeltaT + 1) * m_cols + i_dT * m_cols + i_dT];
				fdmt_type_* arg0 = &piarrOut[iF * (IDeltaT + 1) * m_cols + (i_dT - 1) * m_cols + i_dT];
				
				for (int j = 0; j < (m_cols - i_dT); ++j)
				{
					float t = (b_ones) ? 1.0 : (float)piarrImg[iF * m_cols + j];
					result[j] = (fdmt_type_)((((float)arg0[j]) * ((float)i_dT) + t) / ((float)(i_dT + 1)));
					
				}
			}
	}

}
//-----------------------------------------------------------------
unsigned int CFdmtC::calc_MaxDT(const float val_fmin_MHz, const float val_fmax_MHz, const float length_of_pulse
	, const float val_DM_Max, const int nchan)
{
	float t0 = 4148.8 * (1.0 / (val_fmin_MHz * val_fmin_MHz) -
		1.0 / (val_fmax_MHz * val_fmax_MHz));
	float td = 4148.8 * val_DM_Max * (1.0 / (val_fmin_MHz * val_fmin_MHz) -
		1.0 / (val_fmax_MHz * val_fmax_MHz));
	float t_tel = 1.0E-6 / (val_fmax_MHz - val_fmin_MHz);
	float temp = length_of_pulse / t_tel;
	float val_M = ((td / t_tel) / (temp * temp));
	unsigned int ireturn = (unsigned int)(td / val_M / length_of_pulse);
	return (0 == ireturn) ? 1 : ireturn;
}
//----------------------------------------------
int  CFdmtC::calc_quant_iterations()
{
	int quantMtrx = m_nchan;
	int* iarrDEltas = new int[m_nchan];
	float* arrFreq = new float[m_nchan + 1];
	int ideltaT0 = calc_deltaT(m_Fmin, m_Fmin + (m_Fmax - m_Fmin) / m_nchan);
	for (int i = 0; i < m_nchan; ++i)
	{
		iarrDEltas[i] = ideltaT0 + 1;
		arrFreq[i] = m_Fmin + i * (m_Fmax - m_Fmin) / m_nchan;
	}
	arrFreq[m_nchan] = m_Fmax;
	//std::cout << "row sum = : " << ( 1 + ideltaT0) * m_nchan << " ; Memory = " << (1 + ideltaT0) * m_nchan *m_cols<<std::endl;

	int qIter = 0;
	for (qIter = 0; qIter < 100; ++qIter)
	{
		if (quantMtrx == 1)
		{
			std::cout << "rows  = " << calc_deltaT(arrFreq[0], arrFreq[1]) + 1 << std::endl;
			break;
		}
		int isum = 0;
		for (int i = 0; i < quantMtrx / 2; ++i)
		{
			iarrDEltas[i] = calc_deltaT(arrFreq[2 * i], arrFreq[2 * i + 2]) + 1;
			isum += iarrDEltas[i];
			arrFreq[i + 1] = arrFreq[2 * i + 2];
		}
		int it = quantMtrx;

		if (quantMtrx % 2 == 1)
		{
			quantMtrx = quantMtrx / 2 + 1;
			iarrDEltas[quantMtrx - 1] = iarrDEltas[it - 1];
			isum += iarrDEltas[quantMtrx - 1];
			arrFreq[quantMtrx] = m_Fmax;
		}
		else
		{
			quantMtrx = quantMtrx / 2;
		}
		//std::cout << "iter = "<< qIter << "  quant marx =    "<< quantMtrx<< "  row sum = : " << isum << "Memory = " << isum * m_cols << std::endl;
		/*for (int i = 0; i < quantMtrx; ++i)
		{
			std::cout << "iarrDEltas[ " << i << "] = " << iarrDEltas[i] << "  arrFreq[ " <<i<< "] = "<< arrFreq[i]  << std::endl;
		}*/
	}

	delete[]iarrDEltas;
	delete[]arrFreq;
	return qIter;
}

//----------------------------------------------
int  CFdmtC::calc_quant_iterations_and_lengthSubMtrxArray(int ** pparrLength)
{
	
	*pparrLength = (int*)malloc((1 +ceil_power2__(m_nchan +1) )* sizeof(int));
	
	int quantMtrx = m_nchan;
	(*pparrLength)[0] = quantMtrx;
	int* iarrQuantRows = new int[m_nchan];
	float* arrFreq = new float[m_nchan + 1];
	int ideltaT0 = calc_deltaT(m_Fmin, m_Fmin + (m_Fmax - m_Fmin) / m_nchan);
	for (int i = 0; i < m_nchan; ++i)
	{
		iarrQuantRows[i] = ideltaT0 + 1;
		arrFreq[i] = m_Fmin + i * (m_Fmax - m_Fmin) / m_nchan;
	}
	arrFreq[m_nchan] = m_Fmax;
	//std::cout << "row sum = : " << ( 1 + ideltaT0) * m_nchan << " ; Memory = " << (1 + ideltaT0) * m_nchan *m_cols<<std::endl;

	int qIter = 0;
	for (qIter = 0; qIter < 100; ++qIter)
	{
		if (quantMtrx == 1)
		{
			std::cout << "rows  = " << calc_deltaT(arrFreq[0], arrFreq[1]) + 1 << std::endl;
			break;
		}
		int isum = 0;
		for (int i = 0; i < quantMtrx / 2; ++i)
		{
			iarrQuantRows[i] = calc_deltaT(arrFreq[2 * i], arrFreq[2 * i + 2]) + 1;
			isum += iarrQuantRows[i];
			arrFreq[i + 1] = arrFreq[2 * i + 2];
		}
		int it = quantMtrx;

		if (quantMtrx % 2 == 1)
		{
			quantMtrx = quantMtrx / 2 + 1;
			iarrQuantRows[quantMtrx - 1] = iarrQuantRows[it - 1];
			isum += iarrQuantRows[quantMtrx - 1];
			arrFreq[quantMtrx] = m_Fmax;
		}
		else
		{
			quantMtrx = quantMtrx / 2;
		}
		(*pparrLength)[qIter + 1] = quantMtrx;
		//std::cout << "iter = "<< qIter << "  quant marx =    "<< quantMtrx<< "  row sum = : " << isum << "Memory = " << isum * m_cols << std::endl;
		/*for (int i = 0; i < quantMtrx; ++i)
		{
			std::cout << "iarrQuantRows[ " << i << "] = " << iarrQuantRows[i] << "  arrFreq[ " <<i<< "] = "<< arrFreq[i]  << std::endl;
		}*/
	}
	*pparrLength = (int*)realloc((*pparrLength), (qIter + 1) * sizeof(int));
	delete[]iarrQuantRows;
	delete[]arrFreq;
	return qIter;
}

//---------------------------------
int CFdmtC::calc_deltaT(const float f0, const float f1)
{
	return (int)(ceil((1.0 / (f0 * f0) - 1.0 / (f1 * f1)) / (1.0 / (m_Fmin * m_Fmin) - 1.0 / (m_Fmax * m_Fmax)) * (m_imaxDt - 1.0)));
}
//---------------------------------------------------------------------------
void CFdmtC::calcNextStateConfig(const int QuantMtrx, const int* IArrSubmtrxRows, const float* ARrFreq
	, int &quantMtrx, int* iarrSubmtrxRows, float* arrFreq)
{
	int i = 0;
	for ( i = 0; i < QuantMtrx / 2; ++i)
	{		
		iarrSubmtrxRows[i] = calc_deltaT(ARrFreq[2 * i], ARrFreq[2 * i + 2]) + 1;		
		arrFreq[i] = ARrFreq[2 * i];
	}
	i--;
	if (QuantMtrx % 2 == 1)
	{
		quantMtrx = QuantMtrx / 2 + 1;
		arrFreq[quantMtrx - 1] = ARrFreq[2 * i + 2];
		iarrSubmtrxRows[quantMtrx - 1] = IArrSubmtrxRows[QuantMtrx - 1];
	}
	else
	{
		quantMtrx = QuantMtrx / 2;
	}
	arrFreq[quantMtrx] = m_Fmax;
}
//------------------------------------------------------------------------------------------------
void calcCumSum(const int* iarrQntSubmtrxRows, const int quantSubMtrx, int* iarrCumSum)
{
	iarrCumSum[0] = 0;
	for (int i = 1; i < quantSubMtrx; ++i)
	{
		iarrCumSum[i] = iarrCumSum[i - 1] + iarrQntSubmtrxRows[i - 1];
	}
}
//------------------------------------------------------------------------------------------------
void calcCumSum_(const int* iarrQntSubmtrxRows, const int quantSubMtrx, int* iarrCumSum)
{
	iarrCumSum[0] = 0;
	for (int i = 1; i < (1 + quantSubMtrx); ++i)
	{
		iarrCumSum[i] = iarrCumSum[i - 1] + iarrQntSubmtrxRows[i - 1];
	}
}
//-------------------------------------------
unsigned long long ceil_power2__(const unsigned long long n)
{
	unsigned long long irez = 1;
	for (int i = 0; i < 63; ++i)
	{
		if (irez >= n)
		{
			return irez;
		}
		irez = irez << 1;
	}
	return -1;
}
//-------------------------------------------------------------
void CFdmtC::fncFdmtIterationC_old_old(fdmt_type_* p0, int& quantSubMtrx, int* iarrQntSubmtrxRows, float* arrFreq
	, fdmt_type_* p1)
{
	// 1. calculations of current configuration
	int quantSubMtrxCur = -1;
	int* iarrSubmtrxRowsCur = new int[quantSubMtrx];
	float* arrFreqCur = new float[quantSubMtrx + 1];
	calcNextStateConfig(quantSubMtrx, iarrQntSubmtrxRows, arrFreq, quantSubMtrxCur, iarrSubmtrxRowsCur, arrFreqCur);
	// 1!

	// 2. calc. 2 2-dimentional auxillary arrays
	int* piarr_j0 = new int[quantSubMtrxCur * iarrSubmtrxRowsCur[0]];
	int* piarr_j1 = new int[quantSubMtrxCur * iarrSubmtrxRowsCur[0]];
	//
#pragma omp parallel // 
	{
		for (int i = 0; i < quantSubMtrxCur; ++i)
		{
			float coeff0 = fnc_delay(arrFreq[2 * i], arrFreq[2 * i + 1]) / fnc_delay(arrFreq[2 * i], arrFreq[2 * i + 2]);
			float coeff1 = fnc_delay(arrFreq[2 * i + 1], arrFreq[2 * i + 2]) / fnc_delay(arrFreq[2 * i], arrFreq[2 * i + 2]);

			for (int j = 0; j < iarrSubmtrxRowsCur[i]; ++j)
			{
				piarr_j0[i * iarrSubmtrxRowsCur[0] + j] = (int)(coeff0 * ((float)j));
				piarr_j1[i * iarrSubmtrxRowsCur[0] + j] = (int)(coeff1 * ((float)j));
			}
		}
	}
	// 2!
	// 
	// 3. cumulative sums calculation:
	int* iarrCumSum = new int[quantSubMtrx];
	int* iarrCumSumCur = new int[quantSubMtrxCur];
	calcCumSum(iarrQntSubmtrxRows, quantSubMtrx, iarrCumSum);
	calcCumSum(iarrSubmtrxRowsCur, quantSubMtrxCur, iarrCumSumCur);
	//3!

	// 4. combining each 2 neibore submatrices
#pragma omp parallel // 
	{
		for (int i = 0; i < quantSubMtrxCur; ++i)
		{
			fdmt_type_* pout = &p1[iarrCumSumCur[i] * m_cols];
			fdmt_type_* pinp0 = &p0[iarrCumSum[i * 2] * m_cols];
			if ((i * 2 + 1) >= quantSubMtrx)
			{
				memcpy(pout, pinp0, iarrQntSubmtrxRows[quantSubMtrx - 1] * m_cols * sizeof(fdmt_type_));
				break;
			}

			fdmt_type_* pinp1 = &p0[iarrCumSum[i * 2 + 1] * m_cols];
			for (int j = 0; j < iarrSubmtrxRowsCur[i]; ++j)
			{
				int j0 = piarr_j0[i * iarrSubmtrxRowsCur[0] + j];
				int j1 = piarr_j1[i * iarrSubmtrxRowsCur[0] + j];
				for (int k = 0; k < m_cols; ++k)
				{
					pout[j * m_cols + k] = pinp0[j0 * m_cols + k];
					if ((k - j0) >= 0)
					{
						pout[j * m_cols + k] += pinp1[j1 * m_cols + k - j0];
					}
				}

			}
		}
	}
	// 4!

	// 5. refreshing of configuration
	quantSubMtrx = quantSubMtrxCur;
	memcpy(iarrQntSubmtrxRows, iarrSubmtrxRowsCur, quantSubMtrxCur * sizeof(int));
	memcpy(arrFreq, arrFreqCur, (1 + quantSubMtrxCur) * sizeof(float));
	//5!
	delete[]iarrSubmtrxRowsCur;
	delete[] arrFreqCur;
	delete[]piarr_j0;
	delete[]piarr_j1;
	delete[]iarrCumSum;
	delete[]iarrCumSumCur;
	return;
}
//---------------------------------------------------------------------------------------
// ////-------------------------------------------------------------------------
void CFdmtC::process_image_old_old(fdmt_type_* piarrImgInp       // on-device input image	
	, fdmt_type_* piarrImgOut	// OUTPUT image
	, const bool b_ones
)
{
	// 1. quant iteration's calculation
	const int QIt = calc_quant_iterations();
	// !1

	// 2 calc quantity of rows for initialization
	const int ideltaT = calc_deltaT(m_Fmin, m_Fmin + (m_Fmax - m_Fmin) / m_nchan);

	// !2

	// 3. declare pointers 
	fdmt_type_* p0 = 0;
	fdmt_type_* p1 = 0;
	fdmt_type_* piarrOut_0 = 0;
	fdmt_type_* piarrOut_1 = 0;
	// !3

	// 4. allocate memory 

	piarrOut_0 = (fdmt_type_*)calloc(m_nchan * (ideltaT + 1) * m_cols, sizeof(fdmt_type_));
	piarrOut_1 = (fdmt_type_*)calloc(m_nchan * (ideltaT + 1) * m_cols, sizeof(fdmt_type_));
	// !4

	// 5. call initialization func	
	fnc_initC(piarrImgInp, ideltaT, piarrOut_0, b_ones);
	// !5

	// 6.pointers fixing
	p0 = piarrOut_0;
	p1 = piarrOut_1;
	// 6!

	// 7. initialization of configuration parameters
	int quantSubMtrx = m_nchan;
	int* iarrQntSubmtrxRows = new int[m_nchan];
	float* arrFreq = new float[m_nchan + 1];

	for (int i = 0; i < m_nchan; ++i)
	{
		iarrQntSubmtrxRows[i] = ideltaT + 1;
		arrFreq[i] = m_Fmin + i * (m_Fmax - m_Fmin) / m_nchan;
	}
	arrFreq[m_nchan] = m_Fmax;
	// !7

	// 8. calculations
	for (int iit = 0; iit < QIt; ++iit)
	{
		fncFdmtIterationC_old_old(p0, quantSubMtrx, iarrQntSubmtrxRows, arrFreq, p1);

		// exchange order of pointers
		fdmt_type_* pt = p0;
		p0 = p1;
		p1 = pt;
	}
	// !8

	memcpy(piarrImgOut, p0, m_cols * m_imaxDt * sizeof(fdmt_type_));

	delete[]iarrQntSubmtrxRows;
	delete[]arrFreq;

	free(piarrOut_0);
	free(piarrOut_1);
}
////-------------------------------------------------------------------------
void CFdmtC::process_image_old(fdmt_type_* piarrImgInp, fdmt_type_* piarrImgOut, const bool b_ones)
{
	int** pparrRowsCumSum = NULL;
	float** pparrFreq = NULL;
	int* parrQuantMtrx = NULL;
	int iNumIter = -1;

	create_config(pparrRowsCumSum, pparrFreq, &parrQuantMtrx, &iNumIter);
	// 1!

	//2. declare pointers 
	fdmt_type_* p0 = 0;
	fdmt_type_* p1 = 0;
	fdmt_type_* piarrOut_0 = 0;
	fdmt_type_* piarrOut_1 = 0;
	// !2

	// 3. allocate memory 
	piarrOut_0 = (fdmt_type_*)calloc((pparrRowsCumSum[0])[parrQuantMtrx[0]] * m_cols, sizeof(fdmt_type_));
	piarrOut_1 = (fdmt_type_*)calloc((pparrRowsCumSum[1])[parrQuantMtrx[1]] * m_cols, sizeof(fdmt_type_));
	// !3

   // 4. call initialization func
	const int ideltaT = calc_deltaT(m_Fmin, m_Fmin + (m_Fmax - m_Fmin) / m_nchan);
	fnc_initC(piarrImgInp, ideltaT, piarrOut_0, b_ones);
	// !4

	// 5.pointers fixing
	p0 = piarrOut_0;
	p1 = piarrOut_1;
	// 5!

	// 6. calculations
	for (int iit = 1; iit < (1 + iNumIter); ++iit)
	{
		fncFdmtIterationC_old(p0, iit, parrQuantMtrx, pparrFreq, pparrRowsCumSum, p1);

		// exchange order of pointers
		fdmt_type_* pt = p0;
		p0 = p1;
		p1 = pt;
	}
	// !8


	/*free(parrFreq);
	free(parrRowsCumSum);
	free(pparrQuantMtrx);*/

	memcpy(piarrImgOut, p0, m_cols * m_imaxDt * sizeof(fdmt_type_));


	for (int i = 0; i < (iNumIter + 1); ++i)
	{
		free(pparrFreq[i]);
		free(pparrRowsCumSum[i]);
	}

	free(pparrRowsCumSum);
	free(pparrFreq);
	free(parrQuantMtrx);
	free(piarrOut_0);
	free(piarrOut_1);

}
//-----------------------------------------------------------------------------------------
void CFdmtC::fncFdmtIterationC_old(fdmt_type_* p0, const int  iit, int* parrQuantMtrx, float** pparrFreq, int** pparrRowsCumSum, fdmt_type_* p1)
{
	// 1. extract config for previous mtrix (p0 matrix)  
	int quantSubMtrx = parrQuantMtrx[iit - 1]; // quant of submatixes
	int* iarrCumSum = pparrRowsCumSum[iit - 1];
	float* arrFreq = pparrFreq[iit - 1];
	// 1!  

	// 2. extract config for curent matrix (p0 matrix)
	int quantSubMtrxCur = parrQuantMtrx[iit]; // quant of submatixes
	int* iarrCumSumCur = pparrRowsCumSum[iit];
	float* arrFreqCur = pparrFreq[iit];
	// 2! 

	// 3. combining 2  adjacent matrices
	for (int i = 0; i < quantSubMtrxCur; ++i)
	{
		fdmt_type_* pout = &p1[iarrCumSumCur[i] * m_cols];

		fdmt_type_* pinp0 = &p0[iarrCumSum[i * 2] * m_cols];

		if ((i * 2 + 1) >= quantSubMtrx)
		{
			int quantLastSubmtrxRows = iarrCumSum[quantSubMtrx] - iarrCumSum[quantSubMtrx - 1];
			memcpy(pout, pinp0, quantLastSubmtrxRows * m_cols * sizeof(fdmt_type_));
			break;
		}

		fdmt_type_* pinp1 = &p0[iarrCumSum[i * 2 + 1] * m_cols];
		int quantSubtrxRowsCur = iarrCumSumCur[i + 1] - iarrCumSumCur[i];
		float coeff0 = fnc_delay(arrFreq[2 * i], arrFreq[2 * i + 1]) / fnc_delay(arrFreq[2 * i], arrFreq[2 * i + 2]);
		float coeff1 = fnc_delay(arrFreq[2 * i + 1], arrFreq[2 * i + 2]) / fnc_delay(arrFreq[2 * i], arrFreq[2 * i + 2]);
		for (int j = 0; j < quantSubtrxRowsCur; ++j)
		{
			int j0 = (int)(coeff0 * ((float)j));
			int j1 = (int)(coeff1 * ((float)j));

			for (int k = 0; k < m_cols; ++k)
			{
				pout[j * m_cols + k] = pinp0[j0 * m_cols + k];
				if ((k - j0) >= 0)
				{
					pout[j * m_cols + k] += pinp1[j1 * m_cols + k - j0];
				}
			}
		}
	}
	return;
}