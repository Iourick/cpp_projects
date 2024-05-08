#pragma once
#include <cmath>
#include "FdmtB.h"
#include "FdmtCpu.h"
#include "Constants.h"
class CFdmtCpu_float:public CFdmtB
{
public:
	~CFdmtCpu_float();
	CFdmtCpu_float();
	CFdmtCpu_float(const  CFdmtCpu_float& R);
	CFdmtCpu_float& operator=(const CFdmtCpu_float& R);	
	CFdmtCpu_float(
		const float Fmin
		, const float Fmax		
		, int nchan // quant channels/rows of input image, including consisting of zeroes
		, const int cols
		, int imaxDt // quantity of rows of output image
	);
//------------------------------------------------------------------------------------------
	
	
	virtual void process_image(float* piarrImgInp, float* piarrImgOut, const bool b_ones);

	void fncFdmtIterationC(float* p0, const int  iit, float* p1);

	void fnc_initC(float* piarrImg, const int IDeltaT, float* piarrOut, bool b_ones);
	
};
float fnc_delay_h(const float fmin, const float fmax);








