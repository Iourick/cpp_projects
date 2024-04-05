#include "StreamParams.h"
#include <fstream>
#include <iostream>


CStreamParams::~CStreamParams()
{
    if (m_stream)
    {
        fclose(m_stream);
    }

}
//---------------------------------------------------------------------------
CStreamParams::CStreamParams()
{
    m_lenarr = 0;
    m_n_p = 0;
    m_max = 0;
    m_f_min = 0;
    m_f_max = 0;
    m_SigmaBound = 0;
    m_numBegin = 0;
    m_numEnd = 0;
    m_lenChunk = 0;
    m_numCurChunk = 0;
    m_D_max = 0.;
    m_IMaxDT = 0;
    m_stream = nullptr;
}


//---------------------------------------------------------------------------

// конструктор копирования
CStreamParams::CStreamParams(const  CStreamParams& R)
{
    m_lenarr = R.m_lenarr;
    m_n_p = R.m_n_p;
    m_max = R.m_max;
    m_f_min = R.m_f_min;
    m_f_max = R.m_f_max;
    m_SigmaBound = R.m_SigmaBound;
    m_numBegin = R.m_numBegin;
    m_numEnd = R.m_numEnd;
    m_lenChunk = R.m_lenChunk;
    m_numCurChunk = R.m_numCurChunk;
    m_D_max = R.m_D_max;
    m_stream = R.m_stream;
    m_IMaxDT = R.m_IMaxDT;
}

// оператор присваивания
CStreamParams& CStreamParams::operator=(const CStreamParams& R)
{
    if (this == &R)
    {
        return *this;
    }
    m_lenarr = R.m_lenarr;
    m_n_p = R.m_n_p;
    m_max = R.m_max;
    m_f_min = R.m_f_min;
    m_f_max = R.m_f_max;
    m_SigmaBound = R.m_SigmaBound;
    m_numBegin = R.m_numBegin;
    m_numEnd = R.m_numEnd;
    m_lenChunk = R.m_lenChunk;
    m_numCurChunk = R.m_numCurChunk;
    m_D_max = R.m_D_max;
    m_stream = R.m_stream;
    m_IMaxDT = R.m_IMaxDT;
    return *this;
}


//-------------------------------------------------------------------

// парам конструктор 1
CStreamParams::CStreamParams(char* chInpFilePass, const unsigned int numBegin, const unsigned int numEnd
    , const unsigned int lenChunk)
{
    m_stream = fopen(chInpFilePass, "rb");  
    
    fread(&m_lenarr, sizeof(int), 1, m_stream);
    fread(&m_n_p, sizeof(int), 1, m_stream);    
    fread(& m_D_max, sizeof(float), 1, m_stream);
    fread(& m_f_min, sizeof(float), 1, m_stream);
    fread(& m_f_max, sizeof(float), 1, m_stream);
    fread(& m_SigmaBound, sizeof(float), 1, m_stream);
    fread(&m_IMaxDT, sizeof(int), 1, m_stream);
    m_numBegin = numBegin;
    m_numEnd = numEnd;
    m_lenChunk = lenChunk;
    m_numCurChunk = 0;

}
//-------------------------------------------------------------------

