#pragma once
#include <cmath>

#include "Constants.h"
class CFdmtC
{
public:
	~CFdmtC();
	CFdmtC();
	CFdmtC(const  CFdmtC& R);
	CFdmtC& operator=(const CFdmtC& R);	
	CFdmtC(
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
	int** m_pparrRowsCumSum;
	float** m_pparrFreq ;
	int* m_parrQuantMtrx;
	int m_iNumIter;
	
	void process_image(fdmt_type_* piarrImgInp, fdmt_type_* piarrImgOut, const bool b_ones);

	void fncFdmtIterationC(fdmt_type_* p0, const int  iit, fdmt_type_* p1);

	void fnc_initC(fdmt_type_* piarrImg, const int IDeltaT, fdmt_type_* piarrOut, bool b_ones);

	static  unsigned int calc_MaxDT(const float val_fmin_MHz, const float val_fmax_MHz, const float length_of_pulse
		, const float val_DM_Max, const int nchan);

	int  calc_quant_iterations();

	int calc_deltaT(const float f0, const float f1);		

	void calcNextStateConfig(const int QuantMtrx, const int* IArrSubmtrxRows, const float* ARrFreq
		, int& quantMtrx, int* iarrSubmtrxRows, float* arrFreq);		

	int  calc_quant_iterations_and_lengthSubMtrxArray(int** pparrLength);

	void create_config(int**& pparrRowsCumSum, float**& pparrFreq, int** pparrQuantMtrx, int* piNumIter);	
	
	/**************         OLD *******************************************************************************************/
	void process_image_old_old(fdmt_type_* piarrImgInp       // on-device input image	
		, fdmt_type_* piarrImgOut, const bool b_ones);

	void fncFdmtIterationC_old_old(fdmt_type_* p0, int& quantSubMtrx, int* iarrQntSubmtrxRows, float* arrFreq
		, fdmt_type_* p1);

	void process_image_old(fdmt_type_* piarrImgInp       // on-device input image	
		, fdmt_type_* piarrImgOut	// OUTPUT image
		, const bool b_ones);

	void fncFdmtIterationC_old(fdmt_type_* p0, const int  iit, int* parrQuantMtrx, float** pparrFreq, int** pparrRowsCumSum, fdmt_type_* p1);
};

inline double fnc_delay(const float fmin, const float fmax)
{
	return 1.0 / (fmin * fmin) - 1.0 / (fmax * fmax);
}

void calcCumSum(const int* iarrQntSubmtrxRows, const int quantSubMtrx, int* iarrCumSum);

void calcCumSum_(const int* iarrQntSubmtrxRows, const int quantSubMtrx, int* iarrCumSum);

unsigned long long ceil_power2__(const unsigned long long n);




