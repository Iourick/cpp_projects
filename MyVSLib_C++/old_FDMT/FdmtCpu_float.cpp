#include "FdmtCpu_float.h"
#include <math.h>
#include <array>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <omp.h>

CFdmtCpu_float::~CFdmtCpu_float()
{		
}
//---------------------------------------
CFdmtCpu_float::CFdmtCpu_float() :CFdmtB()
{	
}
//-----------------------------------------------------------

CFdmtCpu_float::CFdmtCpu_float(const  CFdmtCpu_float& R) :CFdmtB(R)
{	
}
//-------------------------------------------------------------------
CFdmtCpu_float& CFdmtCpu_float::operator=(const CFdmtCpu_float& R)
{
	if (this == &R)
	{
		return *this;
	}
	CFdmtB:: operator= (R);
	
	return *this;
}

//--------------------------------------------------------------------
CFdmtCpu_float::CFdmtCpu_float(
	  const float Fmin
	, const float Fmax	
	, const int nchan // quant channels/rows of input image, including consisting of zeroes
	, const int cols		
    , const int imaxDt // quantity of rows of output image
): CFdmtB(Fmin	,  Fmax,  nchan ,cols,  imaxDt)
{		
}

////-------------------------------------------------------------------------
void CFdmtCpu_float::process_image(float* piarrImgInp, float* piarrImgOut, const bool b_ones)
{	
	//1. declare pointers 
	float* p0 = 0;
	float* p1 = 0;
	float* piarrOut_0 = 0;
	float* piarrOut_1 = 0;
	// !1
	int num = 100;
	auto start = std::chrono::high_resolution_clock::now();
	// 2. allocate memory 
	if (!(piarrOut_0 = (float*)calloc(m_pparrRowsCumSum_h[0][m_parrQuantMtrx_h[0]] * m_cols, sizeof(float))))
	{
		printf("Can't allocate memory  for piarrOut_0 in  CFdmtCpu_float::process_image(..)");
		return;
	}

	if (!(piarrOut_1 = (float*)calloc((m_pparrRowsCumSum_h[1])[m_parrQuantMtrx_h[1]] * m_cols, sizeof(float))))
	{
		printf("Can't allocate memory  for piarrOut_1 in  CFdmtCpu_float::process_image(..)");
		free(piarrOut_0);
		return;
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	std::cout << "  Time taken by memalloc: " << duration.count() / ((double)num) << " microseconds" << std::endl;
	
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
		fncFdmtIterationC(p0, iit, p1);
		// exchange order of pointers
		float* pt = p0;
		p0 = p1;
		p1 = pt;
	}
	// !5
	memcpy(piarrImgOut, p0, m_cols * m_imaxDt * sizeof(float));	
	free(piarrOut_0);
	free(piarrOut_1);
}

//-----------------------------------------------------------------------------------------
void CFdmtCpu_float::fncFdmtIterationC(float* p0, const int  iit, float* p1)
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
			float* pout = &p1[iarrCumSumCur[i] * m_cols];
			float* pinp0 = &p0[iarrCumSum[i * 2] * m_cols];

			if ((i * 2 + 1) >= quantSubMtrx)
			{
				int quantLastSubmtrxRows = iarrCumSum[quantSubMtrx] - iarrCumSum[quantSubMtrx - 1];

                 #pragma omp parallel for
				for (int j = 0; j < quantLastSubmtrxRows * m_cols; ++j)
				{
					pout[j] = pinp0[j];
				}
				
				break;
			}

			float* pinp1 = &p0[iarrCumSum[i * 2 + 1] * m_cols];
			int quantSubtrxRowsCur = iarrCumSumCur[i + 1] - iarrCumSumCur[i];
			float coeff0 = 1.0f;// fnc_delay_h(arrFreq[2 * i], arrFreq[2 * i + 1]) / fnc_delay_h(arrFreq[2 * i], arrFreq[2 * i + 2]);
			float coeff1 = 1.0f;// fnc_delay_h(arrFreq[2 * i + 1], arrFreq[2 * i + 2]) / fnc_delay_h(arrFreq[2 * i], arrFreq[2 * i + 2]);
#pragma omp parallel for
			for (int j = 0; j < quantSubtrxRowsCur; ++j)
			{

				int j0 = 0;
				for (int k = 0; k < m_cols; ++k)
				{
					pout[j * m_cols + k] = pinp0[m_cols + k];
					if ((k - j0) >= 0)
					{
						pout[j * m_cols + k] += pinp1[m_cols + k - j0];
					}
				}
			}
		
		}
	// 3!


	return;
}
//--------------------------------------------------------------------------------------

void  CFdmtCpu_float::fnc_initC(float* piarrImg, const int IDeltaT, float* piarrOut, bool b_ones )
{

	//memset(piarrOut, 0, m_nchan * m_cols * (IDeltaT + 1) * sizeof(float ));
	

#pragma omp parallel for// 
			for (int i = 0; i < m_nchan; ++i)
			{
				int num0 = i * (IDeltaT + 1) * m_cols;
				int num1 = i * m_cols;

				for (int j = 0; j < m_cols; ++j)
				{
					piarrOut[num0 + j] = piarrImg[num1 + j];
				}				
				
			}
			
	
#pragma omp parallel for// OMP (
	
		for (int i_dT = 1; i_dT < (IDeltaT + 1); ++i_dT)
			for (int iF = 0; iF < m_nchan; ++iF)
			{
				float* result = &piarrOut[iF * (IDeltaT + 1) * m_cols + i_dT * m_cols + i_dT];
				float* arg0 = &piarrOut[iF * (IDeltaT + 1) * m_cols + (i_dT - 1) * m_cols + i_dT];
				
				for (int j = 0; j < (m_cols - i_dT); ++j)
				{
					
					result[j] = (((arg0[j]) * ((float)i_dT) + piarrImg[iF * m_cols + j]) / ((float)(i_dT + 1)));
					
				}
			}
	

}
float fnc_delay_h(const float fmin, const float fmax)
{
	return (1.0 / (fmin * fmin) - 1.0 / (fmax * fmax));
}

