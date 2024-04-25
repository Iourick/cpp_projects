#pragma once
#include <cmath>
#include "FdmtB.h"
#include "FdmtCpu.h"
#include "Constants.h"
class CFdmtCpu:public CFdmtB
{
public:
	~CFdmtCpu();
	CFdmtCpu();
	CFdmtCpu(const  CFdmtCpu& R);
	CFdmtCpu& operator=(const CFdmtCpu& R);	
	CFdmtCpu(
		const float Fmin
		, const float Fmax		
		, int nchan // quant channels/rows of input image, including consisting of zeroes
		, const int cols
		, int imaxDt // quantity of rows of output image
	);
//------------------------------------------------------------------------------------------
	
	
	virtual void process_image(fdmt_type_* piarrImgInp, fdmt_type_* piarrImgOut, const bool b_ones);

	void fncFdmtIterationC(fdmt_type_* p0, const int  iit, fdmt_type_* p1);

	void fnc_initC(fdmt_type_* piarrImg, const int IDeltaT, fdmt_type_* piarrOut, bool b_ones);
	
};
float fnc_delay_h(const float fmin, const float fmax);








