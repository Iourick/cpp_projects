//#pragma once
#include <vector>
class CFragment
{
public:
	//~CFragment();
	CFragment();
	CFragment(const  CFragment& R);
	CFragment& operator=(const CFragment& R);
	CFragment(const float* parr, const int IDim, const int NBeginRow
		, const int NBeginCol, const int NWidth, const float VAlsnr, const float cohDisp);

	
	//---------------------------------------------------------------------------
	// beginning row of the fragment in numeration of base raster
	int m_nBeginRow;
	// beginning column of the fragment in numeration of base raster
	int m_nBeginCol;
	// dimention of fragment raster m_dim x m_dim
	int m_dim;
	// array with data, length = m_dim x m_dim
	std::vector<float> m_vctData;
	// window width
	int m_width;
	// achived SNR
	float m_snr;
	// coherent dispersion
	float m_cohDisp;
};