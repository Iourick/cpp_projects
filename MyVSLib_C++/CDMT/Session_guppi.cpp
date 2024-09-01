#include "Session_guppi.h"
//-------------------------------------------
CSession_guppi::CSession_guppi() :CSessionB()
{
}

//--------------------------------------------
CSession_guppi::CSession_guppi(const  CSession_guppi& R) :CSessionB(R)
{
}

//-------------------------------------------
CSession_guppi& CSession_guppi::operator=(const CSession_guppi& R)
{
    if (this == &R)
    {
        return *this;
    }
    CSessionB:: operator= (R);

    return *this;
}

//--------------------------------- 
CSession_guppi::CSession_guppi(const char* strGuppiPath, const char* strOutPutPath, const float t_p
    , const double d_min, const double d_max, const float sigma_bound, const int length_sum_wnd, const int nbin, const int nfft)
    :CSessionB(strGuppiPath, strOutPutPath, t_p
        , d_min, d_max, sigma_bound, length_sum_wnd, nbin, nfft)
{
}
//------------------------------------
int CSession_guppi::calcQuantBlocks(unsigned long long* pilength)
{
    FILE* rb_File = fopen(m_strInpPath, "rb");
    if (!rb_File)
    {
        printf("Can't open input file for block calculation ");
        return -1;
    }

    int nbits = 0;
    float chanBW = 0;
    int npol = 0;
    bool bdirectIO = 0;
    float centfreq = 0;
    int nchan = 0;
    float obsBW = 0;
    long long nblocksize = 0;
    EN_telescope TELESCOP = GBT;
    int ireturn = 0;
    float tresolution = 0.;
    *pilength = 0;
    for (int i = 0; i < 1 << 26; ++i)
    {       
        std::int64_t pos0 = ftell(rb_File);
        if (!readTelescopeHeader(
            rb_File
            , &nbits
            , &chanBW
            , &npol
            , &bdirectIO
            , &centfreq
            , &nchan
            , &obsBW
            , &nblocksize
            , &TELESCOP
            , &tresolution
        )
            )
        {
            break;
        }

        ireturn++;
        (*pilength) += (unsigned long)nblocksize;
        unsigned long long ioffset = (unsigned long)nblocksize;
       // std::cout << "i = " << i << " ; nblocksize = " << nblocksize << " (*pilength) = " << (*pilength) << std::endl;
        if (bdirectIO)
        {
            unsigned long num = (ioffset + 511) / 512;
            ioffset = num * 512;
        }

        fseek(rb_File, ioffset, SEEK_CUR);

    }
    fclose(rb_File);
    return ireturn;
}

//------------------------------------------------------------
// //--------------------------------------------------------------
void  CSession_guppi::shift_file_pos(FILE** prb_File, const int IShift)
{
    fseek(prb_File[0], IShift, SEEK_CUR);
}
//----------------------------------------------------
bool CSession_guppi::openFileReadingStream(FILE**& prb_File)
{
    FILE* rb_File = fopen(m_strInpPath, "rb");
    if (!rb_File)
    {
        printf("Can't open RAW file for reading");
        return false;
    }
    prb_File[0] = rb_File;
    return true;
}

//---------------------------------------------
bool CSession_guppi::readTelescopeHeader(FILE* r_file
    , int* nbits
    , float* chanBW
    , int* npol
    , bool* bdirectIO
    , float* centfreq
    , int* nchan
    , float* obsBW
    , long long* nblocksize
    , EN_telescope* TELESCOP
    , float* tresolution
)
{
    //1. download enough data
    char strHeader[MAX_HEADER_LENGTH] = { 0 };
    //fgets(strHeader, sizeof(strHeader), r_file);
    size_t sz = fread(strHeader, sizeof(char), MAX_HEADER_LENGTH, r_file);

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

    // 11.downloading BLOCKSIZE
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
//----------------------------------------------------------------------
bool  CSession_guppi::createCurrentTelescopeHeader(FILE** prb_File)
{
    int nbits = 0;
    float chanBW = 0;
    int npol = 0;
    bool bdirectIO = 0;
    float centfreq = 0;
    int nchan = 0;
    float obsBW = 0;
    long long nblocksize = 0;
    EN_telescope TELESCOP = GBT;
    float tresolution = 0.;
    if (!readTelescopeHeader(
        *prb_File
        , &nbits
        , &chanBW
        , &npol
        , &bdirectIO
        , &centfreq
        , &nchan
        , &obsBW
        , &nblocksize
        , &TELESCOP
        , &tresolution
    )
        )
    {
        return false;
    }

    // 1!
    // 2. creating a current TelescopeHeader
    m_header = CTelescopeHeader(
        nbits
        , chanBW
        , npol
        , bdirectIO
        , centfreq
        , nchan
        , obsBW
        , nblocksize
        , TELESCOP
        , tresolution
    );
    // 2!

    return true;
}


//------------------------------------------
size_t  CSession_guppi::download_chunk(FILE** rb_File, char* d_parrInput, const long long QUantDownloadingBytes)
{
    const long long position0 = ftell(*rb_File);
    long long quantDownloadingBytesPerChannel = QUantDownloadingBytes / m_header.m_nchan;
    long long quantTotalBytesPerChannel = m_header.m_nblocksize / m_header.m_nchan;
    //	
    char* p = (m_header.m_bSraightchannelOrder) ? d_parrInput : d_parrInput + (m_header.m_nchan - 1) * quantDownloadingBytesPerChannel;
    size_t sz_rez = 0;
    for (int i = 0; i < m_header.m_nchan; ++i)
    {
       // long long position1 = ftell(*rb_File);
        sz_rez += fread(p, sizeof(char), quantDownloadingBytesPerChannel, *rb_File);
      //  long long position2 = ftell(*rb_File);
        if (m_header.m_bSraightchannelOrder)
        {
            p += quantDownloadingBytesPerChannel;
        }
        else
        {
            p -= quantDownloadingBytesPerChannel;
        }

        if (i < m_header.m_nchan - 1)
        {
            fseek(*rb_File, quantTotalBytesPerChannel - quantDownloadingBytesPerChannel, SEEK_CUR);
        }

       // long long position3 = ftell(*rb_File);
    }
   // long long position4 = ftell(*rb_File);
    fseek(*rb_File, -(m_header.m_nchan - 1) * quantTotalBytesPerChannel, SEEK_CUR);
   // long long position5 = ftell(*rb_File);
    return sz_rez;
}
//-----------------------------------------------------------------------
//------------------------------------
bool CSession_guppi::navigateToBlock(FILE* rb_File, const int IBlockNum)
{
    const long long position = ftell(rb_File);
    int nbits = 0;
    float chanBW = 0;
    int npol = 0;
    bool bdirectIO = 0;
    float centfreq = 0;
    int nchan = 0;
    float obsBW = 0;
    long long nblocksize = 0;
    EN_telescope TELESCOP = GBT;
    float tresolution = 0.;

    for (int i = 0; i < IBlockNum; ++i)
    {
        long long pos0 = ftell(rb_File);
        if (!CTelescopeHeader::readGuppiHeader(
            rb_File
            , &nbits
            , &chanBW
            , &npol
            , &bdirectIO
            , &centfreq
            , &nchan
            , &obsBW
            , &nblocksize
            , &TELESCOP
            , &tresolution
        ))
        {
            fseek(rb_File, position, SEEK_SET);
            return false;
        }
        if (i == (IBlockNum - 1))
        {
            m_header = CTelescopeHeader(
                nbits
                , chanBW
                , npol
                , bdirectIO
                , centfreq
                , nchan
                , obsBW
                , nblocksize
                , TELESCOP
                , tresolution
            );
            // 2!               
            return true;
        }


        unsigned long long ioffset = (unsigned long)nblocksize;

        if (bdirectIO)
        {
            unsigned long num = (ioffset + 511) / 512;
            ioffset = num * 512;
        }

        fseek(rb_File, ioffset, SEEK_CUR);

    }

    return true;
}

bool  CSession_guppi::unpack_chunk(const long long lenChunk, const int j, inp_type_* d_parrInput, void* pcmparrRawSignalCur)
{
    return false;
}

//---------------------------------------------------------------------------
void CSession_guppi::freeInputMemory(void* parrInput, void* pcmparrRawSignalCur)
{
}
//----------------------------------------------------------------------
bool CSession_guppi::allocateInputMemory(void** parrInput, const int QUantDownloadingBytesForChunk, void** pcmparrRawSignalCur
    , const int QUantChunkComplexNumbers)
{
    
    return true;
}
//----------------------------------------------------------

void CSession_guppi::createChunk(CChunkB** ppchunk
    , const float Fmin
    , const float Fmax
    , const int npol
    , const int nchan   
    , const unsigned int len_sft
    , const int Block_id
    , const int Chunk_id
    , const double d_max
    , const double d_min
    , const int ncoherent
    , const float sigma_bound
    , const int length_sum_wnd
    , const int nbin
    , const int nfft
    , const int noverlap
    , const float tsamp)
{
}
//----------------------------------------------------------------
bool CSession_guppi::closeFileReadingStream(FILE**& prb_File)
{
    fclose(prb_File[0]);
    free(prb_File);
    return true;
}
//----------------------------------------------------------
void CSession_guppi::rewindFilePos(FILE** prb_File, const int  QUantTotalChannelBytes)
{
    // rewind to the beginning of the data block
    fseek(*prb_File, -QUantTotalChannelBytes, SEEK_CUR);
    unsigned long long ioffset = m_header.m_nblocksize;

    if (m_header.m_bdirectIO)
    {
        unsigned long long num = (ioffset + 511) / 512;
        ioffset = num * 512;
    }

    fseek(*prb_File, ioffset, SEEK_CUR);
}
//------------------------------------------------------------------
//------------------------------------------------------------------
size_t  CSession_guppi::calc_ShiftingBytes(unsigned int QUantChunkBytes)
{
    return (size_t)(QUantChunkBytes / m_header.m_nchan);
}