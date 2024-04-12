#include "FdmtB.h"

#include <math.h>
#include <array>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

CFdmtB::~CFdmtB()
{
	if (m_iNumIter > 0)
	{
		for (int i = 0; i < (m_iNumIter + 1); ++i)
		{
			if (m_pparrFreq[i])
			{
				free(m_pparrFreq[i]);
				m_pparrFreq[i] = NULL;
			}

			if (m_pparrRowsCumSum[i])
			{
				free(m_pparrRowsCumSum[i]);
			}
			m_pparrRowsCumSum[i] = NULL;
		}
		free(m_pparrFreq);
		m_pparrFreq = NULL;
		free(m_pparrRowsCumSum);
		m_pparrRowsCumSum = NULL;
		if (m_parrQuantMtrx)
		{
			free(m_parrQuantMtrx);
		}
		m_parrQuantMtrx = NULL;
	}
	m_iNumIter = 0;
}
//---------------------------------------
CFdmtB::CFdmtB()
{
	m_Fmin = 0;
	m_Fmax = 0;
	m_nchan = 0;
	m_cols = 0;
	m_imaxDt = 0;
	m_pparrRowsCumSum = NULL;
	m_pparrFreq = NULL;
	m_parrQuantMtrx = NULL;
	m_iNumIter = 0;
}
//-----------------------------------------------------------

CFdmtB::CFdmtB(const  CFdmtB& R) :CFdmtB()
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
CFdmtB& CFdmtB::operator=(const CFdmtB& R)
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
CFdmtB::CFdmtB(
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

	create_config(m_pparrRowsCumSum, m_pparrFreq, &m_parrQuantMtrx, &m_iNumIter);
}

////-------------------------------------------------------------------------
void CFdmtB::process_image(fdmt_type_* piarrImgInp, fdmt_type_* piarrImgOut, const bool b_ones)
{	
}
//------------------------------------------------------------------------------
void  CFdmtB::create_config(int**& pparrRowsCumSum, float**& pparrFreq, int** pparrQuantMtrx, int* piNumIter)
{
	// 1. calculation iterations quanttity *piNumIter and array *pparrQuantMtrx of quantity submatrices for each iteration
	// *pparrQuantMtrx  has length = *piNumIter +1
	// (*pparrQuantMtrx  has length)[0] = m_nchan , for initialization
	*piNumIter = calc_quant_iterations_and_lengthSubMtrxArray(pparrQuantMtrx);
	// 1!

	// 2. memory allocation for 2 auxillary arrays
	int* iarrQuantMtrx = *pparrQuantMtrx;

	pparrFreq = (float**)malloc((*piNumIter + 1) * sizeof(float*)); // Allocate memory for m pointers to int

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


//-----------------------------------------------------------------
unsigned int CFdmtB::calc_MaxDT(const float val_fmin_MHz, const float val_fmax_MHz, const float length_of_pulse
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
int  CFdmtB::calc_quant_iterations()
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
			//std::cout << "rows  = " << calc_deltaT(arrFreq[0], arrFreq[1]) + 1 << std::endl;
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
int  CFdmtB::calc_quant_iterations_and_lengthSubMtrxArray(int** pparrLength)
{

	*pparrLength = (int*)malloc((1 + ceil_power2__(m_nchan + 1)) * sizeof(int));

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
			//std::cout << "rows  = " << calc_deltaT(arrFreq[0], arrFreq[1]) + 1 << std::endl;
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
int CFdmtB::calc_deltaT(const float f0, const float f1)
{
	return (int)(ceil((1.0 / (f0 * f0) - 1.0 / (f1 * f1)) / (1.0 / (m_Fmin * m_Fmin) - 1.0 / (m_Fmax * m_Fmax)) * (m_imaxDt - 1.0)));
}
//---------------------------------------------------------------------------
void CFdmtB::calcNextStateConfig(const int QuantMtrx, const int* IArrSubmtrxRows, const float* ARrFreq
	, int& quantMtrx, int* iarrSubmtrxRows, float* arrFreq)
{
	int i = 0;
	for (i = 0; i < QuantMtrx / 2; ++i)
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
void CFdmtB::calcCumSum(const int* iarrQntSubmtrxRows, const int quantSubMtrx, int* iarrCumSum)
{
	iarrCumSum[0] = 0;
	for (int i = 1; i < quantSubMtrx; ++i)
	{
		iarrCumSum[i] = iarrCumSum[i - 1] + iarrQntSubmtrxRows[i - 1];
	}
}
//------------------------------------------------------------------------------------------------
void CFdmtB::calcCumSum_(const int* iarrQntSubmtrxRows, const int quantSubMtrx, int* iarrCumSum)
{
	iarrCumSum[0] = 0;
	for (int i = 1; i < (1 + quantSubMtrx); ++i)
	{
		iarrCumSum[i] = iarrCumSum[i - 1] + iarrQntSubmtrxRows[i - 1];
	}
}
//-------------------------------------------
unsigned long long CFdmtB::ceil_power2__(const unsigned long long n)
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
//------------------------------------------------
size_t CFdmtB::calcSizeAuxBuff_fdmt_()
{
	return 0;
}
//---------------------
size_t  CFdmtB::calc_size_input()
{
	return m_cols * m_nchan * sizeof(fdmt_type_);
}
//---------------------
size_t CFdmtB::calc_size_output()
{
	return m_cols * m_imaxDt * sizeof(fdmt_type_);
}

