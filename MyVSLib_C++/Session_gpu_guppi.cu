#include "Session_gpu_guppi.cuh"
#include <string>
#include "stdio.h"
#include <iostream>

#include "OutChunk.h"
#include <cufft.h>
#include "FdmtU.cuh"
#include <stdlib.h>
#include "Fragment.cuh"
#include "helper_functions.h"
#include "helper_cuda.h"
#include "yr_cart.h"
#include "Chunk.cuh"
#include <complex>
#include "Session_gpu.cuh"

//-------------------------------------------
CSession_gpu_guppi::CSession_gpu_guppi():CSession_gpu()
{
}

//--------------------------------------------
CSession_gpu_guppi::CSession_gpu_guppi(const  CSession_gpu_guppi& R) :CSession_gpu(R)
{
}

//-------------------------------------------
CSession_gpu_guppi& CSession_gpu_guppi::operator=(const CSession_gpu_guppi& R)
{
    if (this == &R)
    {
        return *this;
    }
    CSession_gpu:: operator= (R);   

    return *this;
}

//--------------------------------- 
CSession_gpu_guppi::CSession_gpu_guppi(const char* strGuppiPath, const char* strOutPutPath, const float t_p
    , const double d_max, const float sigma_bound, const int length_sum_wnd)
    :CSession_gpu(strGuppiPath,  strOutPutPath,  t_p, d_max, sigma_bound,  length_sum_wnd)
{   
}
//------------------------------------
int CSession_gpu_guppi::calcQuantBlocks( unsigned long long* pilength)
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
//-----------------------
float CSession_gpu_guppi::fncTest(float val)
{
    float ff = 10.0;
    return 21 * val;
}
//-------------------------------------------
// After return 
// file cursor is installed on beginning of data block
bool CSession_gpu_guppi::readTelescopeHeader(FILE* r_file
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
//----------------------------------------------------
bool CSession_gpu_guppi::openFileReadingStream(FILE**& prb_File)
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
//----------------------------------------------------------------
 bool CSession_gpu_guppi::closeFileReadingStream(FILE**& prb_File)
{
     fclose(prb_File[0]);
    
    return true;
}
//-------------------------------------------------------------
bool CSession_gpu_guppi::createCurrentTelescopeHeader(FILE** prb_File)
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
void CSession_gpu_guppi::download_and_unpack_chunk(FILE** prb_File, const long long lenChunk, const int j
    , inp_type_* d_parrInput, cufftComplex* pcmparrRawSignalCur)
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

    const dim3 blockSize = dim3(1024, 1, 1);
    const dim3 gridSize = dim3((lenChunk + blockSize.x - 1) / blockSize.x, m_header.m_nchan, 1);
    unpackInput << < gridSize, blockSize >> > (pcmparrRawSignalCur, (inp_type_*)d_parrInput, lenChunk, m_header.m_nchan, m_header.m_npol);
    cudaDeviceSynchronize();

    /*std::vector <std::complex<float>> data4(lenChunk* m_header.m_nchan * m_header.m_npol/2, 0);
    cudaMemcpy(data4.data(), pcmparrRawSignalCur, lenChunk * m_header.m_nchan * m_header.m_npol / 2 * sizeof(cufftComplex),
        cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();*/
}
//-----------------------------------------------------------------
void CSession_gpu_guppi::rewindFilePos(FILE**prb_File, const int  QUantTotalChannelBytes)
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

//------------------------------------------
size_t  CSession_gpu_guppi::downloadChunk(FILE* rb_File, char* d_parrInput, const long long QUantDownloadingBytes)
{
    const long long position0 = ftell(rb_File);
    long long quantDownloadingBytesPerChannel = QUantDownloadingBytes / m_header.m_nchan;
    long long quantTotalBytesPerChannel = m_header.m_nblocksize / m_header.m_nchan;
    //	
    char* p = (m_header.m_bSraightchannelOrder) ? d_parrInput : d_parrInput + (m_header.m_nchan - 1) * quantDownloadingBytesPerChannel;
    size_t sz_rez = 0;
    for (int i = 0; i < m_header.m_nchan; ++i)
    {
        long long position1 = ftell(rb_File);
        sz_rez += fread(p, sizeof(char), quantDownloadingBytesPerChannel, rb_File);
        long long position2 = ftell(rb_File);
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
            fseek(rb_File, quantTotalBytesPerChannel - quantDownloadingBytesPerChannel, SEEK_CUR);
        }

        long long position3 = ftell(rb_File);
    }
    long long position4 = ftell(rb_File);
    fseek(rb_File, -(m_header.m_nchan - 1) * quantTotalBytesPerChannel, SEEK_CUR);
    long long position5 = ftell(rb_File);
    return sz_rez;
}