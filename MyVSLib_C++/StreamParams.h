#pragma once
#include <fstream>

class CStreamParams
{
public:
    ~CStreamParams();

    CStreamParams();

    CStreamParams(const  CStreamParams& R);

    CStreamParams& operator=(const CStreamParams& R);

    CStreamParams(char* chInpFilePass, const unsigned int numBegin, const unsigned int numEnd
        , const unsigned int lenChunk);

    unsigned int m_lenarr;
    unsigned int m_n_p;
    float m_max; 
    float m_f_min;
    float m_f_max;
    float m_D_max;
    float m_SigmaBound;
    unsigned int m_numBegin;
    unsigned int m_numEnd;
    unsigned int m_lenChunk;
    unsigned int m_numCurChunk;
    unsigned int m_IMaxDT;
    FILE* m_stream;         
        
};

