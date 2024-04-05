#include "Fragment.h"
#include <stdlib.h>
#include <iostream>

//CFragment::~CFragment()
//{
//    if (m_pvctData)
//    {
//        delete m_pvctData;
//    }
//    m_pvctData = NULL;
//}
//-------------------------------------------
CFragment::CFragment()
{
     m_nBeginRow = 0;    
     m_nBeginCol = 0;
    // dimention of fragment raster m_dim x m_dim
     m_dim = 0;
    // array with data, length = m_dim x m_dim
     m_vctData = std::vector<float>(1);
    // window width
     m_width = 0;
    // achived SNR
     m_snr = 0.;
     m_cohDisp = 0.;
}

//--------------------------------------------
CFragment::CFragment(const  CFragment& R)
{
    m_nBeginRow = R.m_nBeginRow;
    m_nBeginCol = R.m_nBeginCol;    
    m_dim = R.m_dim;
    m_width = R.m_width;
    m_snr = R.m_snr;
    m_vctData = R.m_vctData;
    m_cohDisp = R.m_cohDisp;
}

//-------------------------------------------
CFragment& CFragment::operator=(const CFragment& R)
{
    if (this == &R)
    {
        return *this;
    }
    m_nBeginRow = R.m_nBeginRow;
    m_nBeginCol = R.m_nBeginCol;
    m_dim = R.m_dim;
    m_width = R.m_width;
    m_snr = R.m_snr;
    m_vctData = R.m_vctData;
    m_cohDisp = R.m_cohDisp;
    return *this;
}

//--------------------------------- 
CFragment::CFragment(const float* parr, const int IDim, const int NBeginRow
    , const int NBeginCol, const int NWidth, const float VAlsnr, const float cohDisp)
{
    m_nBeginRow = NBeginRow;
    m_nBeginCol = NBeginCol;
    m_dim = IDim;
    m_width = NWidth;
    m_snr = VAlsnr;
    
    std::vector<float> floatVector(parr, parr + m_dim * m_dim);      
    m_vctData = floatVector;
    m_cohDisp = cohDisp;    
}
//--------------------------------- 
