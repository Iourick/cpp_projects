#include "FdmtCpu.h"
#include <math.h>
#include <array>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <omp.h>

int NUmOperation = 0;
CFdmtCpu::~CFdmtCpu()
{	
	if (m_pparrFreq_h != NULL)
	{
		for (int i = 0; i < (m_iNumIter + 1); ++i)
		{
			if (m_pparrFreq_h[i] != NULL)
			{
				free(m_pparrFreq_h[i]);
				m_pparrFreq_h[i] = NULL;
			}
		}
		free(m_pparrFreq_h);
	}


	if (m_pparrRowsCumSum_h != NULL)
	{
		for (int i = 0; i < (m_iNumIter + 1); ++i)
		{
			if (m_pparrRowsCumSum_h[i] != NULL)
			{
				free(m_pparrRowsCumSum_h[i]);
				m_pparrRowsCumSum_h[i] = NULL;
			}
		}
		free(m_pparrRowsCumSum_h);
	}

	if (m_parrQuantMtrx_h != NULL)
	{
		free(m_parrQuantMtrx_h);
		m_parrQuantMtrx_h = NULL;
	}

	m_iNumIter = 0;
}
//---------------------------------------
CFdmtCpu::CFdmtCpu() :CFdmtB()
{	
	m_pparrRowsCumSum_h = NULL;
	m_pparrFreq_h = NULL;
	m_parrQuantMtrx_h = NULL;
	m_iNumIter = 0;
}
//-----------------------------------------------------------

CFdmtCpu::CFdmtCpu(const  CFdmtCpu& R) :CFdmtB(R)
{	
	m_iNumIter = R.m_iNumIter;

	m_parrQuantMtrx_h = (int*)malloc((R.m_iNumIter + 1) * sizeof(int));
	memcpy(m_parrQuantMtrx_h, R.m_parrQuantMtrx_h, (R.m_iNumIter + 1) * sizeof(int));

	m_pparrFreq_h = (float**)malloc((R.m_iNumIter + 1) * sizeof(float*));
	for (int i = 0; i < (R.m_iNumIter + 1); ++i)
	{
		m_pparrFreq_h[i] = (float*)malloc((1 + R.m_parrQuantMtrx_h[i]) * sizeof(float));
		memcpy(m_pparrFreq_h[i], R.m_pparrFreq_h[i], (1 + R.m_parrQuantMtrx_h[i]) * sizeof(float));
	}

	m_pparrRowsCumSum_h = (int**)malloc((R.m_iNumIter + 1) * sizeof(int*));
	for (int i = 0; i < (R.m_iNumIter + 1); ++i)
	{
		m_pparrRowsCumSum_h[i] = (int*)malloc((1 + R.m_parrQuantMtrx_h[i]) * sizeof(int));
		memcpy(m_pparrRowsCumSum_h[i], R.m_pparrRowsCumSum_h[i], (1 + R.m_parrQuantMtrx_h[i]) * sizeof(int));
	}
}
//-------------------------------------------------------------------
CFdmtCpu& CFdmtCpu::operator=(const CFdmtCpu& R)
{
	if (this == &R)
	{
		return *this;
	}
	CFdmtB:: operator= (R);
	m_iNumIter = R.m_iNumIter;

	if (m_pparrFreq_h != NULL)
	{
		free(m_pparrFreq_h);
	}

	m_pparrFreq_h = (float**)malloc((R.m_iNumIter + 1) * sizeof(float*));
	if (m_pparrFreq_h != NULL)
	{
		for (int i = 0; i < (R.m_iNumIter + 1); ++i)
		{
			m_pparrFreq_h[i] = (float*)malloc((1 + R.m_parrQuantMtrx_h[i]) * sizeof(float));
			memcpy(m_pparrFreq_h[i], R.m_pparrFreq_h[i], (1 + R.m_parrQuantMtrx_h[i]) * sizeof(float));
		}
	}

	if (m_pparrRowsCumSum_h != NULL)
	{
		free(m_pparrRowsCumSum_h);
	}

	m_pparrRowsCumSum_h = (int**)malloc((R.m_iNumIter + 1) * sizeof(int*));
	if (m_pparrRowsCumSum_h != NULL)
	{
		for (int i = 0; i < (R.m_iNumIter + 1); ++i)
		{
			m_pparrRowsCumSum_h[i] = (int*)malloc((1 + R.m_parrQuantMtrx_h[i]) * sizeof(int));
			memcpy(m_pparrRowsCumSum_h[i], R.m_pparrRowsCumSum_h[i], (1 + R.m_parrQuantMtrx_h[i]) * sizeof(int));
		}
	}

	if (m_parrQuantMtrx_h != NULL)
	{
		free(m_parrQuantMtrx_h);
		m_parrQuantMtrx_h = NULL;
	}
	m_parrQuantMtrx_h = (int*)malloc((R.m_iNumIter + 1) * sizeof(int));
	memcpy(m_parrQuantMtrx_h, R.m_parrQuantMtrx_h, (R.m_iNumIter + 1) * sizeof(int));
	
	return *this;
}

//--------------------------------------------------------------------
CFdmtCpu::CFdmtCpu(
	  const float Fmin
	, const float Fmax	
	, const int nchan // quant channels/rows of input image, including consisting of zeroes
	, const int cols		
    , const int imaxDt // quantity of rows of output image
): CFdmtB(Fmin	,  Fmax,  nchan ,cols,  imaxDt)
{	

	create_config(m_pparrRowsCumSum_h, m_pparrFreq_h, &m_parrQuantMtrx_h, &m_iNumIter);
}
//------------------------------------------------------------------------------
void  CFdmtCpu::create_config(int**& pparrRowsCumSum, float**& pparrFreq, int** pparrQuantMtrx, int* piNumIter)
{
	// 1. calculation iterations quanttity *piNumIter and array *pparrQuantMtrx of quantity submatrices for each iteration
	// *pparrQuantMtrx  has length = *piNumIter +1
	// (*pparrQuantMtrx  has length)[0] = m_nchan , for initialization
	*piNumIter = calc_quant_iterations_and_lengthSubMtrxArray(pparrQuantMtrx);
	// 1!

	// 2. memory allocation for 2 auxillary arrays
	int* iarrQuantMtrx = *pparrQuantMtrx;

	pparrFreq = (float**)malloc((*piNumIter + 1) * sizeof(float*)); //

	pparrRowsCumSum = (int**)malloc((*piNumIter + 1) * sizeof(int*));

	for (int i = 0; i < (*piNumIter + 1); ++i)
	{
		pparrFreq[i] = (float*)malloc((iarrQuantMtrx[i] + 1) * sizeof(float));
		pparrRowsCumSum[i] = (int*)malloc((iarrQuantMtrx[i] + 1) * sizeof(int));
	}
	// 2!

	// 3. initialization 0 step	 
	float* arrFreq = pparrFreq[0];

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

////-------------------------------------------------------------------------
void CFdmtCpu::process_image(fdmt_type_* piarrImgInp, fdmt_type_* piarrImgOut, const bool b_ones)
{	
	//1. declare pointers 
	fdmt_type_* p0 = 0;
	fdmt_type_* p1 = 0;
	fdmt_type_* piarrOut_0 = 0;
	fdmt_type_* piarrOut_1 = 0;
	// !1

	// 2. allocate memory 
	if (!(piarrOut_0 = (fdmt_type_*)calloc(m_pparrRowsCumSum_h[0][m_parrQuantMtrx_h[0]] * m_cols, sizeof(fdmt_type_))))
	{
		printf("Can't allocate memory  for piarrOut_0 in  CFdmtCpu::process_image(..)");
		return;
	}

	if (!(piarrOut_1 = (fdmt_type_*)calloc((m_pparrRowsCumSum_h[1])[m_parrQuantMtrx_h[1]] * m_cols, sizeof(fdmt_type_))))
	{
		printf("Can't allocate memory  for piarrOut_1 in  CFdmtCpu::process_image(..)");
		free(piarrOut_0);
		return;
	}

	
	// !2
	
   // 3. call initialization func
	const int ideltaT = calc_deltaT(m_Fmin, m_Fmin + (m_Fmax - m_Fmin) / m_nchan);

	NUmOperation = 0;
	fnc_initC(piarrImgInp, ideltaT, piarrOut_0, b_ones);
	printf("NUmOperation = %i\n", NUmOperation);
	// !3

	// 4.pointers fixing
	p0 = piarrOut_0;
	p1 = piarrOut_1;
	// 4!

	// 5. calculations
	for (int iit = 1; iit < (1 +m_iNumIter); ++iit)
	{		
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

//-----------------------------------------------------------------------------------------
void CFdmtCpu::fncFdmtIterationC(fdmt_type_* p0, const int  iit, fdmt_type_* p1)
{
	// 1. extract config for previous mtrix (p0 matrix)  
	int quantSubMtrx = m_parrQuantMtrx_h[iit - 1]; // quant of submatixes
	int* iarrCumSum = m_pparrRowsCumSum_h[iit - 1];
	float* arrFreq = m_pparrFreq_h[iit - 1];
	// 1!  

	// 2. extract config for curent matrix (p0 matrix)
	int quantSubMtrxCur = m_parrQuantMtrx_h[iit]; // quant of submatixes
	int* iarrCumSumCur = m_pparrRowsCumSum_h[iit];
	float* arrFreqCur = m_pparrFreq_h[iit];
	// 2! 

	// 3. combining 2  adjacent matrices

	
	for (int i = 0; i < quantSubMtrxCur; ++i)
	{
		fdmt_type_* pout = &p1[iarrCumSumCur[i] * m_cols];
		fdmt_type_* pinp0 = &p0[iarrCumSum[i * 2] * m_cols];

		if ((i * 2 + 1) >= quantSubMtrx)
		{
			int quantLastSubmtrxRows = iarrCumSum[quantSubMtrx] - iarrCumSum[quantSubMtrx - 1];
#pragma omp parallel for
			for (int j = 0; j < quantLastSubmtrxRows * m_cols; ++j)
			{
				pout[j] = pinp0[j];
			}
			//memcpy(pout, pinp0, quantLastSubmtrxRows * m_cols * sizeof(fdmt_type_));
			break;
		}

		fdmt_type_* pinp1 = &p0[iarrCumSum[i * 2 + 1] * m_cols];
		int quantSubtrxRowsCur = iarrCumSumCur[i + 1] - iarrCumSumCur[i];
		float coeff0 = fnc_delay_h(arrFreq[2 * i], arrFreq[2 * i + 1]) / fnc_delay_h(arrFreq[2 * i], arrFreq[2 * i + 2]);
		float coeff1 = fnc_delay_h(arrFreq[2 * i + 1], arrFreq[2 * i + 2]) / fnc_delay_h(arrFreq[2 * i], arrFreq[2 * i + 2]);

#pragma omp parallel for
		
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
	// 3!
	return;
}
//--------------------------------------------------------------------------------------

void  CFdmtCpu::fnc_initC(fdmt_type_* piarrImg, const int IDeltaT, fdmt_type_* piarrOut, bool b_ones )
{

	//memset(piarrOut, 0, m_nchan * m_cols * (IDeltaT + 1) * sizeof(fdmt_type_ ));
	if (!b_ones)
	{
#pragma omp parallel for// 
		
			for (int i = 0; i < m_nchan; ++i)
			{
				for (int j = 0; j < m_cols; ++j)
				{
					piarrOut[i * (IDeltaT + 1) * m_cols + j] = piarrImg[i * m_cols + j];
					++NUmOperation;
				}
					//memcpy(&piarrOut[i * (IDeltaT + 1) * m_cols], &piarrImg[i * m_cols]
					//	, m_cols * sizeof(fdmt_type_));
				
			}
		
	}
	else
	{
		float temp = 1.;
		fdmt_type_ t = (fdmt_type_)temp;

#pragma omp parallel for// 		
			for (int i = 0; i < m_nchan; ++i)
			{
				for (int j=0; j < m_cols;++j)
				{

					piarrOut[i * (IDeltaT + 1) * m_cols + j] = t;
					
				}
			}
		
	}
#pragma omp parallel for// OMP (
	 
		for (int i_dT = 1; i_dT < (IDeltaT + 1); ++i_dT)
			for (int iF = 0; iF < m_nchan; ++iF)
			{
				fdmt_type_* result = &piarrOut[iF * (IDeltaT + 1) * m_cols + i_dT * m_cols + i_dT];
				fdmt_type_* arg0 = &piarrOut[iF * (IDeltaT + 1) * m_cols + (i_dT - 1) * m_cols + i_dT];
				
				for (int j = 0; j < (m_cols - i_dT); ++j)
				{
					float t = (b_ones) ? 1.0 : (float)piarrImg[iF * m_cols + j];
					result[j] = (fdmt_type_)((((float)arg0[j]) * ((float)i_dT) + t) / ((float)(i_dT + 1)));
					++NUmOperation;
				}
			}
}
//------------------------------------------------------------------------------
float fnc_delay_h(const float fmin, const float fmax)
{
	return (1.0 / (fmin * fmin) - 1.0 / (fmax * fmax));
}

