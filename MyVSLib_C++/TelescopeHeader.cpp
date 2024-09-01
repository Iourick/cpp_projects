#include "TelescopeHeader.h"

#include <string.h>
#include <stdlib.h>

#include "stdio.h"
#include <iostream>

CTelescopeHeader::~CTelescopeHeader() = default;

CTelescopeHeader::CTelescopeHeader() 
{
	
	m_nbits = 0;	
	m_chanBW = 0.;	
	m_npol = 0;	
	m_bdirectIO = true;	
    m_centfreq = 0.;	
	m_nchan = 0;	
	m_obsBW = 0.;	
	m_nblocksize = 0.;	
	m_TELESCOP = GBT;
	m_tresolution = 0;
    m_bSraightchannelOrder = true;
}


CTelescopeHeader::CTelescopeHeader(const  CTelescopeHeader& R) 
{
	m_nbits = R.m_nbits;

	m_chanBW = R.m_chanBW;

	m_npol = R.m_npol;

	m_bdirectIO = R.m_bdirectIO;

	m_centfreq = R.m_centfreq;

	m_nchan = R.m_nchan;

	m_obsBW = R.m_obsBW;

	m_nblocksize = R.m_nblocksize;

	m_TELESCOP = R.m_TELESCOP;

	m_tresolution = R.m_tresolution;

    m_bSraightchannelOrder = R.m_bSraightchannelOrder;
}

// 
CTelescopeHeader& CTelescopeHeader::operator=(const CTelescopeHeader& R)
{
    if (this == &R)
    {
        return *this;
    }  

	m_nbits = R.m_nbits;

	m_chanBW = R.m_chanBW;

	m_npol = R.m_npol;

	m_bdirectIO = R.m_bdirectIO;

	m_centfreq = R.m_centfreq;

	m_nchan = R.m_nchan;

	m_obsBW = R.m_obsBW;

	m_nblocksize = R.m_nblocksize;

	m_TELESCOP = R.m_TELESCOP;

	m_tresolution = R.m_tresolution;

    m_bSraightchannelOrder = R.m_bSraightchannelOrder;

    return *this;
}


// 
CTelescopeHeader::CTelescopeHeader(
      const int nbits
    , const float chanBW
    , const int npol
    , const bool bdirectIO
    , const float centfreq
    , const int nchan
    , const float obsBW
    , const int nblocksize
    , const  EN_telescope TELESCOP
    , const float  tresolution
)

{
    m_nbits = nbits;

    if (chanBW < 0.)
    {
        m_chanBW = -chanBW;
        m_bSraightchannelOrder = false;
    }
    else
    {
        m_chanBW = chanBW;
        m_bSraightchannelOrder = true;
    }   

    m_npol = npol;

    m_bdirectIO = bdirectIO;

    m_centfreq = fabs(centfreq);

    m_nchan = nchan;

    m_obsBW = fabs(obsBW);

    m_nblocksize = nblocksize;

    m_TELESCOP = TELESCOP;

    m_tresolution = tresolution;

}

// 
//CTelescopeHeader::CTelescopeHeader(
//    const int nbits
//    , const float chanBW
//    , const int npol
//    , const bool bdirectIO
//    , const float centfreq
//    , const int nchan
//    , const float obsBW
//    , const int nblocksize
//    , const  EN_telescope TELESCOP
//    , const float  tresolution
//    , const bool bSraightchannelOrder
//)
//
//{
//    m_nbits = nbits;
//
//    m_chanBW = chanBW;
//
//    m_npol = npol;
//
//    m_bdirectIO = bdirectIO;
//
//    m_centfreq = centfreq;
//
//    m_nchan = nchan;
//
//    m_obsBW = obsBW;
//
//    m_nblocksize = nblocksize;
//
//    m_TELESCOP = TELESCOP;
//
//    m_tresolution = tresolution;
//
//    m_bSraightchannelOrder = bSraightchannelOrder;
//
//}
//-------------------------------------------------------------------
// After return 
// file cursor is installed on beginning of data block
bool CTelescopeHeader::readGuppiHeader(FILE* r_file
    , int* nbits
    , float* chanBW
    , int* npol
    , bool* bdirectIO
    , float* centfreq
    , int* nchan
    , float* obsBW
    , long long* nblocksize
    , EN_telescope* TELESCOP
    , float *tresolution
)
{    
    //1. download enough data
    char strHeader[MAX_HEADER_LENGTH] = { 0 };
    //fgets(strHeader, sizeof(strHeader), r_file);
    size_t sz = fread(strHeader,sizeof(char), MAX_HEADER_LENGTH, r_file);
    
    if (sz < MAX_HEADER_LENGTH)
    {
        return false;
    }
    // !

    //2. check up mode. if mode != RAW return false    
    if (NULL == strstr(strHeader, "RAW"))
    {
        return false;
    }
    // 2!

    // 3. find 3-rd occurence of "END"
    char* pEND = strHeader;
    for (int i = 0; i < 3; ++i)
    {
        if (NULL == (pEND = strstr(pEND, "END")))
        {
            return false;
        }
        pEND++;
    }
    pEND--;
    long long ioffset = pEND - strHeader;
    // 3!

    // 4.downloading m_bdirectIO
    char* pio = strstr(strHeader, "DIRECTIO");
    if (NULL == pio)
    {
        return false;
    }
    int i_io = atoi(pio + 9);
    *bdirectIO = (i_io == 0) ? false : true;
    //4 !  

    // 5. alignment cursors to beginning of raw data
    ioffset += 3;
    if ((*bdirectIO))
    {
        int num = (ioffset + 511) / 512;
        ioffset = num * 512;
    }
    
    fseek(r_file, ioffset - MAX_HEADER_LENGTH, SEEK_CUR);
   
    // 5!

    // 6.downloading NBITS
    pio = strstr(strHeader, "NBITS");
    if (NULL == pio)
    {
        return false;
    }
    *nbits = atoi(pio + 9);
    //6 ! 

    // 7.downloading CHAN_BW
    pio = strstr(strHeader, "CHAN_BW");
    if (NULL == pio)
    {
        return false;
    }
    *chanBW = atof(pio + 9);
    //7 ! 

    // 8.downloading OBSFREQ
    pio = strstr(strHeader, "OBSFREQ");
    if (NULL == pio)
    {
        return false;
    }
    *centfreq = atof(pio + 9);
    //8 !

    // 9.downloading OBSNCHAN
    pio = strstr(strHeader, "OBSNCHAN");
    if (NULL == pio)
    {
        return false;
    }
    *nchan = atoi(pio + 9);
    //9 !

    // 10.downloading OBSNCHAN
    pio = strstr(strHeader, "OBSBW");
    if (NULL == pio)
    {
        return false;
    }
    *obsBW = atof(pio + 9);
    //10 !

    // 11.downloading BLOCSIZE
    pio = strstr(strHeader, "BLOCSIZE");
    if (NULL == pio)
    {
        return false;
    }
    *nblocksize = atoi(pio + 9);
    //11 !    

    // 12.downloading OBSNCHAN
    pio = strstr(strHeader, "TELESCOP");
    if (NULL == pio)
    {
        return false;
    }
    pio += 9;
    char* pt = strstr(pio, "GBT");
    char* pt1 = NULL;
    *TELESCOP = GBT;
    if (NULL == pt)
    {
        pt = strstr(pio, "PARKES");
        if (NULL == pt)
        {
            return false;
        }
        if ((pt - pio) > 20)
        {
            return false;
        }
        *TELESCOP = PARKES;
    }
    else
    {
        if ((pt - pio) > 20)
        {
            return false;
        }
    }

    //12 !

    // 13.downloading NPOL
    pio = strstr(strHeader, "NPOL");
    if (NULL == pio)
    {
        return false;
    }
    *npol = atoi(pio + 9);
    //13 !

    // 14.downloading time resolution
    pio = strstr(strHeader, "TBIN");
    if (NULL == pio)
    {
        return false;
    }
    *tresolution = atof(pio + 10);
    
    return true;
}
