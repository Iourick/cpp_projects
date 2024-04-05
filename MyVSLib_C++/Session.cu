#include "Session.cuh"
#include <string>
#include "stdio.h"
#include <iostream>
#include "Block.cuh"
#include "OutChunk.h"
#include <cufft.h>
#include "fdmtU_cu.cuh"
#include <stdlib.h>
#include "Fragment.cuh"
 

CSession::~CSession()
{    
    if (m_rbFile)
    {
        fclose(m_rbFile);
    }
    m_rbFile = NULL;
    if (m_wb_file)
    {
        fclose(m_wb_file);
    }
    m_wb_file = NULL;

    if (m_pvctSuccessHeaders)
    {
        delete m_pvctSuccessHeaders;
    }

}
//-------------------------------------------
CSession::CSession()
{    
    m_rbFile = NULL;
    memset(m_strGuppiPath, 0, MAX_PATH_LENGTH * sizeof(char));
    memset(m_strOutPutPath, 0, MAX_PATH_LENGTH * sizeof(char));
    m_pvctSuccessHeaders = new std::vector<COutChunkHeader>();
    m_header = CTelescopeHeader();
    m_t_p = 1.0E-6;
    m_d_max = 0.;
    m_sigma_bound = 10.;
    m_length_sum_wnd = 10;
}

//--------------------------------------------
CSession::CSession(const  CSession& R)
{      
    if (m_rbFile)
    {
        fclose(m_rbFile);
    }
    m_rbFile = R.m_rbFile;
    memcpy(m_strGuppiPath, R.m_strGuppiPath, MAX_PATH_LENGTH * sizeof(char));
    memcpy(m_strOutPutPath, R.m_strOutPutPath, MAX_PATH_LENGTH * sizeof(char));
    if (m_pvctSuccessHeaders)
    {
        m_pvctSuccessHeaders = R.m_pvctSuccessHeaders;
    }
    m_header = R.m_header;  
    m_t_p = R.m_t_p;
    m_d_max = R.m_d_max;
    m_sigma_bound = R.m_sigma_bound;
    m_length_sum_wnd = R.m_length_sum_wnd;
}

//-------------------------------------------
CSession& CSession::operator=(const CSession& R)
{
    if (this == &R)
    {
        return *this;
    }
    
    if (m_rbFile)
    {
        fclose(m_rbFile);
    }
    m_rbFile = R.m_rbFile;
    memcpy(m_strGuppiPath, R.m_strGuppiPath, MAX_PATH_LENGTH * sizeof(char));
    memcpy(m_strOutPutPath, R.m_strOutPutPath, MAX_PATH_LENGTH * sizeof(char));
    if (m_pvctSuccessHeaders)
    {
        m_pvctSuccessHeaders = R.m_pvctSuccessHeaders;
    }
    m_header = R.m_header;    
    m_t_p = R.m_t_p;
    m_d_max = R.m_d_max;
    m_sigma_bound = R.m_sigma_bound;
    m_length_sum_wnd = R.m_length_sum_wnd;
    return *this;
}

//--------------------------------- 
CSession::CSession(const char* strGuppiPath, const char* strOutPutPath, const float t_p
,const double d_max, const float sigma_bound, const int length_sum_wnd)
{
    strcpy(m_strOutPutPath, strOutPutPath);
    strcpy(m_strGuppiPath, strGuppiPath);
    m_rbFile = fopen(strGuppiPath, "rb");    
    m_wb_file = fopen(strOutPutPath, "wb");
    m_pvctSuccessHeaders = new std::vector<COutChunkHeader>();
    m_t_p = t_p;
    m_d_max = d_max;
    m_sigma_bound = sigma_bound;
    m_length_sum_wnd = length_sum_wnd;
}
//------------------------------------
int CSession::calcQuantRemainBlocks(unsigned long long* pilength)
{
    const long long position = ftell(m_rbFile);
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
    *pilength = 0;
    for (int i = 0; i < 1 << 26; ++i)
    {        
        long long pos0 = ftell(m_rbFile);
        if (!CTelescopeHeader::readHeader(
            m_rbFile
            , &nbits
            , &chanBW
            , &npol
            , &bdirectIO
            , &centfreq
            , &nchan
            , &obsBW
            , &nblocksize
            , &TELESCOP
        )
            )
        {
            break;
        }      

        ireturn++;
        (*pilength) += (unsigned long)nblocksize;
        unsigned long long ioffset = (unsigned long)nblocksize;
        std::cout << "i = " << i << " ; nblocksize = " << nblocksize << " (*pilength) = " << (*pilength) << std::endl;
        if (bdirectIO)
        {
            unsigned long num = (ioffset + 511) / 512;
            ioffset = num * 512;
        }

        fseek(m_rbFile, ioffset, SEEK_CUR);     
 
    }
    fseek(m_rbFile, position, SEEK_SET);
    return ireturn;
}

//---------------------------------------------------------------
int CSession::launch()
{
    // calc quantity sessions
    unsigned long long ilength = -1;

    // 1. blocks quantity calculation
    unsigned long long position = ftell(m_rbFile);
    const int IBlock = calcQuantRemainBlocks(&ilength);
    
    fseek(m_rbFile, position, SEEK_SET);
    //!1


    // 2.allocation memory for parametrs in CPU
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
    // !2

    // 3. Performing a loop using the variable nS, nS = 0,..,IBlock. 
    //IBlock - number of bulks
    
    for (int nS = 0; nS < IBlock; ++nS)
    {       
        // 3.1. reading info from current bulk header
        // After return 
        // file cursor is installed on beginning of data block
        if (!CTelescopeHeader::readHeader(
            m_rbFile
            , &nbits
            , &chanBW
            , &npol
            , &bdirectIO
            , &centfreq
            , &nchan
            , &obsBW
            , &nblocksize
            , &TELESCOP
        )
            )
        {            
            return -1;
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
        );
        // 2!

        // calculate N_p
        const int len_sft = calc_len_sft(fabs(m_header.m_chanBW));

        // calculate lenChunk along time axe
        const unsigned int LenChunk = calcLenChunk(len_sft);
        //


        const bool bCHannel_order = (m_header.m_chanBW > 0.) ? true : false;
        CBlock* pBlock = new CBlock(
              m_header.m_centfreq - fabs(m_header.m_chanBW) * m_header.m_nchan / 2.
            , m_header.m_centfreq + fabs(m_header.m_chanBW) * m_header.m_nchan / 2.
            , m_header.m_npol
            , m_header.m_nblocksize
            , m_header.m_nchan            
            , LenChunk
            , len_sft
            , nS
            , m_header.m_nbits
            , bCHannel_order
            , m_d_max
            , m_sigma_bound
            , m_length_sum_wnd 

        );

        //int quantSuccessChunks = 0;

        pBlock->process(m_rbFile , m_pvctSuccessHeaders);      
                
        delete pBlock;
        
        unsigned long long ioffset = m_header.m_nblocksize;
        
        if (bdirectIO)
        {
            unsigned long long num = (ioffset + 511) / 512;
            ioffset = num * 512;
        }

        fseek(m_rbFile, ioffset, SEEK_CUR);  
    }
    
    return 0;
}
//-------------------------------------------
long long CSession::calcLenChunk(const int n_p)
{
    const int N_p = n_p * m_header.m_nchan;
    unsigned int IMaxDT = n_p * m_header.m_nchan;
    float fmin = m_header.m_centfreq - fabs(m_header.m_chanBW) * m_header.m_nchan / 2.;
    float fmax = m_header.m_centfreq + fabs(m_header.m_chanBW) * m_header.m_nchan / 2.;
    const int  IDeltaT = calc_IDeltaT(N_p, fmin, fmax, IMaxDT);

    float valNominator = TOtal_GPU_Bytes - N_p * (sizeof(float) + sizeof(int)
        / 2 + 3 * (IDeltaT + 1) * sizeof(int));

    float valDenominator = m_header.m_nchan * m_header.m_npol / 2 * sizeof(cufftComplex)        
        + 2 * (IDeltaT + 1) * m_header.m_nchan * sizeof(fdmt_type_)
        + 3 * m_header.m_nchan * m_header.m_npol * sizeof(cufftComplex) / 2 
        + 2 * m_header.m_nchan * sizeof(float);
    float tmax = valNominator / valDenominator;
    float treal = (float)(m_header.m_nblocksize * 8 / m_header.m_nchan / m_header.m_npol / m_header.m_nbits );
    float t = (tmax < treal) ? tmax : treal;

    return  pow(2, floor(log2(t)));
}

//-----------------------------------------------------------------
void CSession::writeReport()
{
    char arrch[2000] = { 0 };
    char charrTemp[200] = { 0 };
    for (int i = 0; i < m_pvctSuccessHeaders->size(); ++i)
    {
        memset(charrTemp, 0, 200 * sizeof(char));
        (*m_pvctSuccessHeaders)[i].createOutStr(charrTemp);
        strcat(arrch, charrTemp);
        memset(charrTemp, 0, 200 * sizeof(char));
        sprintf(charrTemp, ", Length of pulse= %.10e", m_t_p);
        strcat(arrch, charrTemp);
        strcat(arrch, "\n");
            //createOutStr(char* pstr)
    }
    size_t elements_written = fwrite(arrch, sizeof(char), strlen(arrch), m_wb_file);
}
//-------------------------------------------------
bool CSession::read_outputlogfile_line(const char *pstrPassLog
    , const int NUmLine
    , int* pnumBlock
    , int* pnumChunk    
    , int* pn_fdmtRows 
    , int* n_fdmtCols
    , int* psucRow
    , int* psucCol
    , int* pwidth
    , float *pcohDisp
    , float* snr
                            )
{
    //1. download enough data
    char line[300] = { 0 };     
    
    FILE* fp = fopen(pstrPassLog, "r");
    if (!fp)
    {
        fprintf(stderr, "Error opening file '%s'\n", pstrPassLog);
        return EXIT_FAILURE;
    }

    /* Get the first line of the file. */
    for (int i = 0; i < NUmLine + 1; ++i)
    {
        fgets(line, 300, fp);
    }
    fclose(fp);
    //2. check up mode. if mode != RAW return false  
    char* p = strstr(line, "Block=");
    if (NULL == p)
    {
        delete p;
        return false;
    }    
    *pnumBlock = atoi(p + 8);

    p = strstr(line, "Chunk=");
    if (NULL == p)
    {
        return false;
    }
    
    *pnumChunk = atoi(p + 8);

    p = strstr(line, "Rows=");
    if (NULL == p)
    {
        delete p;
        return false;
    }
    *pn_fdmtRows = atoi(p + 7);

    p = strstr(line, "Cols=");
    if (NULL == p)
    {
        return false;
    }
    *n_fdmtCols = atoi(p + 7);

    p = strstr(line, "SucRow=");
    if (NULL == p)
    {
        delete p;
        return false;
    }
    *psucRow = atoi(p + 9);

    p = strstr(line, "SucCol=");
    if (NULL == p)
    {
        delete p;
        return false;
    }
    *psucCol = atoi(p + 9);

    p = strstr(line, "SNR=");
    if (NULL == p)
    {
        delete p;
        return false;
    }
    *snr = atof(p + 6); 

    p = strstr(line, "CohDisp=");
    if (NULL == p)
    {
        delete p;
        return false;
    }
    *pcohDisp = atof(p + 10);

    p = strstr(line, "windWidth=");
    if (NULL == p)
    {
        delete p;
        return false;
    }
    *pwidth = atoi(p + 12);    
    return true;
}
//---------------------------------------------------------
bool CSession::analyzeChunk(const COutChunkHeader outChunkHeader,CFragment* pFRg)
{
    if (!navigateToBlock(outChunkHeader.m_numBlock +1))
    {
        return false;
    }

    // calculate N_p
    const int len_sft = calc_len_sft(fabs(m_header.m_chanBW));

    

    // calculate lenChunk along time axe
    const unsigned int LEnChunk = calcLenChunk(len_sft);
    //
    const bool bCHannel_order = (m_header.m_chanBW > 0.) ? true : false;
    CBlock* pBlock = new CBlock(
        m_header.m_centfreq - fabs(m_header.m_chanBW) * m_header.m_nchan / 2.
        , m_header.m_centfreq + fabs(m_header.m_chanBW) * m_header.m_nchan / 2.
        , m_header.m_npol
        , m_header.m_nblocksize
        , m_header.m_nchan
        , LEnChunk
        , len_sft
        , outChunkHeader.m_numBlock
        , m_header.m_nbits
        , bCHannel_order
        , m_d_max
        , m_sigma_bound
        , m_length_sum_wnd

    );

    if (!pBlock->detailedChunkProcessing(m_rbFile, outChunkHeader, pFRg))
    {
        delete pBlock;
        return false;
    }
    delete pBlock;
    return true;
}
//-----------------------------------------------------------------
//------------------------------------
bool CSession::navigateToBlock(const int IBlockNum)
{
    const long long position = ftell(m_rbFile);
    int nbits = 0;
    float chanBW = 0;
    int npol = 0;
    bool bdirectIO = 0;
    float centfreq = 0;
    int nchan = 0;
    float obsBW = 0;
    long long nblocksize = 0;
    EN_telescope TELESCOP = GBT;
    
    for (int i = 0; i < IBlockNum; ++i)
    {
        long long pos0 = ftell(m_rbFile);
        if (!CTelescopeHeader::readHeader(
            m_rbFile
            , &nbits
            , &chanBW
            , &npol
            , &bdirectIO
            , &centfreq
            , &nchan
            , &obsBW
            , &nblocksize
            , &TELESCOP
        ))
        {
            fseek(m_rbFile, position, SEEK_SET);
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

        fseek(m_rbFile, ioffset, SEEK_CUR);

    }
    
    return true;
}

