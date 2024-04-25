#pragma once
#pragma once
#include <cmath>

#include "Constants.h"
class CFdmtB
{
public:
	~CFdmtB();
	CFdmtB();
	CFdmtB(const  CFdmtB& R);
	CFdmtB& operator=(const CFdmtB& R);
	CFdmtB(
		const float Fmin
		, const float Fmax
		, int nchan // quant channels/rows of input image, including consisting of zeroes
		, const int cols
		, int imaxDt // quantity of rows of output image
	);
	//------------------------------------------------------------------------------------------
	int m_nchan; // quant channels/rows of input image, including consisting of zeroes	
	int m_cols;  // quant cols of input image (time axe)
	float m_Fmin;
	float m_Fmax;
	int m_imaxDt; // quantity of rows of output image
	// configuration params:
	int** m_pparrRowsCumSum_h;
	float** m_pparrFreq_h;
	int* m_parrQuantMtrx;
	int m_iNumIter;

	virtual void process_image(fdmt_type_* piarrImgInp, fdmt_type_* piarrImgOut, const bool b_ones);

	

	static  unsigned int calc_MaxDT(const float val_fmin_MHz, const float val_fmax_MHz, const float length_of_pulse
		, const float val_DM_Max, const int nchan);

	int  calc_quant_iterations();

	int calc_deltaT(const float f0, const float f1);

	void calcNextStateConfig(const int QuantMtrx, const int* IArrSubmtrxRows, const float* ARrFreq
		, int& quantMtrx, int* iarrSubmtrxRows, float* arrFreq);

	int  calc_quant_iterations_and_lengthSubMtrxArray(int** pparrLength);

	void create_config(int**& pparrRowsCumSum, float**& pparrFreq, int** pparrQuantMtrx, int* piNumIter);

	virtual size_t calcSizeAuxBuff_fdmt_();

	size_t  calc_size_input();

	size_t  calc_size_output();

	static void calcCumSum(const int* iarrQntSubmtrxRows, const int quantSubMtrx, int* iarrCumSum);

	static void calcCumSum_(const int* iarrQntSubmtrxRows, const int quantSubMtrx, int* iarrCumSum);

	static unsigned long long ceil_power2__(const unsigned long long n);

};

//inline double fnc_delay(const float fmin, const float fmax)
//{
//	return 1.0 / (fmin * fmin) - 1.0 / (fmax * fmax);
//}








