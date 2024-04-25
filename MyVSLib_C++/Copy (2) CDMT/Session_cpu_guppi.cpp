#include "Session_cpu_guppi.h"
#include <string>
#include "stdio.h"
#include <iostream>

#include "OutChunk.h"


#include <stdlib.h>
#include "Fragment.h"
#include <fftw3.h>
#include <hdf5.h> 
#include <complex>
#include "yr_cart.h"
#include "CChunk_cpu.h"
#include <complex>
#include "SessionB.h"


//-------------------------------------------
CSession_cpu_guppi::CSession_cpu_guppi():CSessionB()
{
}

//--------------------------------------------
CSession_cpu_guppi::CSession_cpu_guppi(const  CSession_cpu_guppi& R) :CSessionB(R)
{
}

//-------------------------------------------
CSession_cpu_guppi& CSession_cpu_guppi::operator=(const CSession_cpu_guppi& R)
{
    if (this == &R)
    {
        return *this;
    }
    CSessionB:: operator= (R);   

    return *this;
}

//--------------------------------- 
CSession_cpu_guppi::CSession_cpu_guppi(const char* strGuppiPath, const char* strOutPutPath, const float t_p
    , const double d_min, const double d_max, const float sigma_bound, const int length_sum_wnd, const int nbin, const int nfft)
    :CSessionB( strGuppiPath, strOutPutPath,  t_p   ,  d_min, d_max,  sigma_bound,  length_sum_wnd,  nbin, nfft)
{   
}
//------------------------------------
int CSession_cpu_guppi::calcQuantBlocks( unsigned long long* pilength)
{
    FILE* rb_File = fopen(m_strInpPath, "rb");
    if (! rb_File )
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
        if (16 == i)
        {
            int hh = 0;
        }
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
        std::cout << "i = " << i << " ; nblocksize = " << nblocksize << " (*pilength) = " << (*pilength) << std::endl;
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
//----------------------------------------------------
bool CSession_cpu_guppi::openFileReadingStream(FILE**& prb_File)
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

//-------------------------------------------
// After return 
// file cursor is installed on beginning of data block
bool CSession_cpu_guppi::readTelescopeHeader(FILE* r_file
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

//----------------------------------------------------------------
 bool CSession_cpu_guppi::closeFileReadingStream(FILE**& prb_File)
{
     fclose(prb_File[0]);
    
    return true;
}
//-------------------------------------------------------------
bool CSession_cpu_guppi::createCurrentTelescopeHeader(FILE** prb_File)
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
//----------------------------------------------------------

void CSession_cpu_guppi::download_and_unpack_chunk(FILE** prb_File, const long long lenChunk, const int j
    , inp_type_* d_parrInput, void* pcmparrRawSignalCur)
{
    // number of downloding bytes to each chunk:
    const long long QUantChunkBytes = lenChunk * m_header.m_nchan / 8 * m_header.m_npol * m_header.m_nbits;
    // total number of downloding bytes to each channel:
    const long long QUantTotalChannelBytes = m_header.m_nblocksize / m_header.m_nchan;
    // total number of downloading complex numbers of channel:
    const long long QUantChannelComplexNumbers = lenChunk * m_header.m_npol / 2;

    const int NumChunks = (m_header.m_nblocksize + QUantChunkBytes - 1) / QUantChunkBytes;
    
    //
    if (j == (NumChunks - 1))
    {
       // long long position = ftell(*prb_File);
        long long offset = -QUantChunkBytes / m_header.m_nchan * j + QUantTotalChannelBytes - QUantChunkBytes / m_header.m_nchan;
        fseek(*prb_File, offset, SEEK_CUR);
        //long long position1 = ftell(rb_File);
        

    }

   // long long position3 = ftell(*prb_File);

    size_t sz = downloadChunk(*prb_File, (char*)d_parrInput, QUantChunkBytes);


   // long long position4 = ftell(*prb_File);

   

    /*std::vector<inp_type_ > data0(quantDownloadingBytes/4, 0);
    cudaMemcpy(data0.data(), d_parrInput, quantDownloadingBytes / 4 * sizeof(inp_type_ ),
        cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();*/

    fftwf_complex* p = (fftwf_complex*)pcmparrRawSignalCur;
    unpack_input_cpu_guppi(p, (inp_type_*)d_parrInput, lenChunk, m_header.m_nchan, m_header.m_npol);

    /*std::vector <std::complex<float>> data4(lenChunk* m_header.m_nchan * m_header.m_npol/2, 0);
    cudaMemcpy(data4.data(), pcmparrRawSignalCur, lenChunk * m_header.m_nchan * m_header.m_npol / 2 * sizeof(cufftComplex),
        cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();*/
}
//------------------------------------------------------------------
void CSession_cpu_guppi::unpack_input_cpu_guppi(fftwf_complex* pcmparrRawSignalCur, inp_type_* d_parrInput, const int  lenChunk, const int nchan, const int npol)
{

}

//---------------------------------------------------------------------------
void CSession_cpu_guppi::freeInputMemory(void* parrInput, void* pcmparrRawSignalCur)
{
    free(parrInput);
    fftw_free(pcmparrRawSignalCur);
}
//-----------------------------------------------------------------
void CSession_cpu_guppi::rewindFilePos(FILE**prb_File, const int  QUantTotalChannelBytes)
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