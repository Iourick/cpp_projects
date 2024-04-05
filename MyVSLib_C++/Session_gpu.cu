#include "Session_gpu.cuh"
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
 

CSession_gpu::~CSession_gpu()
{ 
    if (m_pvctSuccessHeaders)
    {
        delete m_pvctSuccessHeaders;
    }

}
//-------------------------------------------
CSession_gpu::CSession_gpu()
{    
    
    memset( m_strInpPath, 0, MAX_PATH_LENGTH * sizeof(char));
    memset(m_strOutPutPath, 0, MAX_PATH_LENGTH * sizeof(char));
    m_pvctSuccessHeaders = new std::vector<COutChunkHeader>();
    m_header = CTelescopeHeader();
    m_pulse_length = 1.0E-6;
    m_d_max = 0.;
    m_sigma_bound = 10.;
    m_length_sum_wnd = 10;
    
}

//--------------------------------------------
CSession_gpu::CSession_gpu(const  CSession_gpu& R)
{  
    memcpy( m_strInpPath, R. m_strInpPath, MAX_PATH_LENGTH * sizeof(char));
    memcpy(m_strOutPutPath, R.m_strOutPutPath, MAX_PATH_LENGTH * sizeof(char));
    if (m_pvctSuccessHeaders)
    {
        m_pvctSuccessHeaders = R.m_pvctSuccessHeaders;
    }
    m_header = R.m_header;  
    m_pulse_length = R.m_pulse_length;
    m_d_max = R.m_d_max;
    m_sigma_bound = R.m_sigma_bound;
    m_length_sum_wnd = R.m_length_sum_wnd;
    
}

//-------------------------------------------
CSession_gpu& CSession_gpu::operator=(const CSession_gpu& R)
{
    if (this == &R)
    {
        return *this;
    }   
    
    memcpy( m_strInpPath, R. m_strInpPath, MAX_PATH_LENGTH * sizeof(char));
    memcpy(m_strOutPutPath, R.m_strOutPutPath, MAX_PATH_LENGTH * sizeof(char));
    if (m_pvctSuccessHeaders)
    {
        m_pvctSuccessHeaders = R.m_pvctSuccessHeaders;
    }
    m_header = R.m_header;    
    m_pulse_length = R.m_pulse_length;
    m_d_max = R.m_d_max;
    m_sigma_bound = R.m_sigma_bound;
    m_length_sum_wnd = R.m_length_sum_wnd;
    
    return *this;
}

//--------------------------------- 
CSession_gpu::CSession_gpu(const char* strGuppiPath, const char* strOutPutPath, const float t_p
,const double d_max, const float sigma_bound, const int length_sum_wnd)
{
    strcpy(m_strOutPutPath, strOutPutPath);
    strcpy( m_strInpPath, strGuppiPath);   
    m_pvctSuccessHeaders = new std::vector<COutChunkHeader>();
    m_pulse_length = t_p;
    m_d_max = d_max;
    m_sigma_bound = sigma_bound;
    m_length_sum_wnd = length_sum_wnd;
}
//------------------------------------
int CSession_gpu::calcQuantBlocks(unsigned long long* pilength)
{
    return -1;
}

//---------------------------------------------------------------
int CSession_gpu::launch()
{
    cudaError_t cudaStatus;
 
    // calc quantity sessions
    unsigned long long ilength = 0;

    // 1. blocks quantity calculation
  
    const int IBlock = calcQuantBlocks(&ilength);
    
   
    //!1

    
   

    // 3. cuFFT plans preparations
       // 3.1 reading first header
   



    cufftHandle plan0 = NULL;
    cufftHandle plan1 = NULL;
    int lenChunk = 0;
    char* d_parrInput = NULL;
    cufftComplex* pcmparrRawSignalCur = NULL;
    CFdmtU fdmt;
    void* pAuxBuff_fdmt = NULL;
    fdmt_type_* d_arrfdmt_norm = NULL;
    cufftComplex* pcarrTemp = NULL; //2	
    cufftComplex* pcarrCD_Out = NULL;//3
    cufftComplex* pcarrBuff = NULL;//3
    char * pInpOutBuffFdmt = NULL;
    CChunk* pChunk = new CChunk();
    
    do_plan_and_memAlloc(&lenChunk
        ,&plan0, &plan1, &fdmt,&d_parrInput, &pcmparrRawSignalCur
        ,&pAuxBuff_fdmt,&d_arrfdmt_norm
        ,&pcarrTemp
        ,&pcarrCD_Out
        , &pcarrBuff, &pInpOutBuffFdmt, &pChunk);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus));
        // Handle the error appropriately
    }
  
    // 4. memory allocation in GPU
    // total number of downloding bytes to each chunk:
    const long long QUantChunkBytes = lenChunk * m_header.m_nchan / 8 * m_header.m_npol * m_header.m_nbits;
    // total number of downloding bytes to each channel:
    const long long QUantTotalChannelBytes = m_header.m_nblocksize * m_header.m_nbits / 8 / m_header.m_nchan;
    const long long QUantChunkComplexNumbers = lenChunk * m_header.m_nchan * m_header.m_npol / 2;
    FILE** prb_File = (FILE**)malloc( sizeof(FILE*));
    if (!prb_File) {
        // Handle memory allocation failure
        return 1;
    }
    
    openFileReadingStream(prb_File);
    
    
    for (int nB = 0; nB < IBlock; ++nB)        
    {   
        
        cout << "                               BLOCK=  " << nB <<endl;
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus));
            // Handle the error appropriately
        }

        createCurrentTelescopeHeader(prb_File);
       

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus));
            // Handle the error appropriately
        }
        // number of downloding bytes to each chunk:
        const long long QUantChunkBytes = lenChunk * m_header.m_nchan / 8 * m_header.m_npol * m_header.m_nbits;
        // total number of downloding bytes to each channel:
        const long long QUantTotalChannelBytes = m_header.m_nblocksize  / m_header.m_nchan;
        // total number of downloading complex numbers of channel:
        const long long QUantChannelComplexNumbers = lenChunk * m_header.m_npol / 2;

        const int NumChunks = (m_header.m_nblocksize + QUantChunkBytes - 1) / QUantChunkBytes;


        
        // !6

        // 7. remains of not readed elements in block
        //long long iremainedBytes = m_header.m_nblocksize;
        float val_coherent_d;
        // !7

        

       

        float valSNR = -1;
        int argmaxRow = -1;
        int argmaxCol = -1;
        float coherentDedisp = -1.;

        

        
        for (int j = 0; j < NumChunks; ++j)
        {
           
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus));
                // Handle the error appropriately
            }
           download_and_unpack_chunk( prb_File,  lenChunk,  j, (inp_type_ *)d_parrInput,  pcmparrRawSignalCur);
            long long position4 = ftell(*prb_File);
            /*if (j!= (NumChunks - 1))
            {
                continue;
            }*/
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus));
                // Handle the error appropriately
            }

            /*std::vector<inp_type_ > data0(quantDownloadingBytes/4, 0);
            cudaMemcpy(data0.data(), d_parrInput, quantDownloadingBytes / 4 * sizeof(inp_type_ ),
                cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();*/          
         
            /*std::vector <std::complex<float>> data4(lenChunk* m_header.m_nchan * m_header.m_npol/2, 0);
            cudaMemcpy(data4.data(), pcmparrRawSignalCur, lenChunk * m_header.m_nchan * m_header.m_npol / 2 * sizeof(cufftComplex),
                cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();*/
           
            
            pChunk->set_blockid(nB);
            pChunk->set_chunkid(j);
            
            pChunk->fncChunkProcessing_gpu(pcmparrRawSignalCur
                , pAuxBuff_fdmt
                , pcarrTemp
                , pcarrCD_Out
                , pcarrBuff
                , pInpOutBuffFdmt
                , d_arrfdmt_norm
                , plan0
                , plan1
                , m_pvctSuccessHeaders);            

        }
        
        

        //std::cout << "/*****************************************************/ " << std::endl;
        //std::cout << "/*****************************************************/ " << std::endl;
        //std::cout << "/*****************************************************/ " << std::endl;
        //iTotal_time = duration.count();
        //std::cout << "Total time:                   " << iTotal_time << " microseconds" << std::endl;
        //std::cout << "FDMT time:                    " << iFdmt_time << " microseconds" << std::endl;
        //std::cout << "Read and data transform time: " << iReadTransform_time << " microseconds" << std::endl;
        //std::cout << "FFT time:                     " << iFFT_time << " microseconds" << std::endl;
        //std::cout << "Mean && disp time:            " << iMeanDisp_time << " microseconds" << std::endl;
        //std::cout << "Normalization time:           " << iNormalize_time << " microseconds" << std::endl;

        //std::cout << "/*****************************************************/ " << std::endl;
        //std::cout << "/*****************************************************/ " << std::endl;
        //std::cout << "/*****************************************************/ " << std::endl;	
        rewindFilePos(prb_File,   QUantTotalChannelBytes);      
    }
   closeFileReadingStream(prb_File);
   free(prb_File);
    cudaFree(pcmparrRawSignalCur);
    cudaFree(d_arrfdmt_norm);
    cudaFree(pAuxBuff_fdmt);
    cudaFree(pcarrTemp); //2	
    cudaFree(pcarrCD_Out);//3
    cudaFree(pcarrBuff);//3
    cudaFree(pInpOutBuffFdmt);
    cudaFree(d_parrInput);
    delete pChunk;
    cufftDestroy(plan0);
    cufftDestroy(plan1);
    return 0;
}
//------------------------------------------------------------------
bool CSession_gpu::openFileReadingStream(FILE**& prb_File)
{
    return false;
}

//------------------------------------------------------------------
bool CSession_gpu::closeFileReadingStream(FILE**& prb_File)
{
    return false;
}
//-----------------------------------------------------------------------
// //-----------------------------------------------------------------
void CSession_gpu::rewindFilePos(FILE** prb_File, const int  QUantTotalChannelBytes)
{
}
//------------------------------------------------------------------------
bool CSession_gpu::readTelescopeHeader(FILE* r_File
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
    return false;
}
//-------------------------------------------------------------
bool CSession_gpu::createCurrentTelescopeHeader(FILE** prb_File)
{
    return false;
}
//---------------------------------------------------------
bool CSession_gpu::analyzeChunk(const COutChunkHeader outChunkHeader, CFragment* pFRg)
{
    cudaError_t cudaStatus;

    // calc quantity sessions
    unsigned long long ilength = 0;

    // 1. blocks quantity calculation

    const int IBlock = calcQuantBlocks(&ilength);


    //!1




    // 3. cuFFT plans preparations
       // 3.1 reading first header




    cufftHandle plan0 = NULL;
    cufftHandle plan1 = NULL;
    int lenChunk = 0;
    char* d_parrInput = NULL;
    cufftComplex* pcmparrRawSignalCur = NULL;
    CFdmtU fdmt;
    void* pAuxBuff_fdmt = NULL;
    fdmt_type_* d_arrfdmt_norm = NULL;
    cufftComplex* pcarrTemp = NULL; //2	
    cufftComplex* pcarrCD_Out = NULL;//3
    cufftComplex* pcarrBuff = NULL;//3
    char* pInpOutBuffFdmt = NULL;
    CChunk* pChunk = new CChunk();

    do_plan_and_memAlloc(&lenChunk
        , &plan0, &plan1, &fdmt, &d_parrInput, &pcmparrRawSignalCur
        , &pAuxBuff_fdmt, &d_arrfdmt_norm
        , &pcarrTemp
        , &pcarrCD_Out
        , &pcarrBuff, &pInpOutBuffFdmt, &pChunk);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus));
        // Handle the error appropriately
    }

    // 4. memory allocation in GPU
    // total number of downloding bytes to each chunk:
    const long long QUantChunkBytes = lenChunk * m_header.m_nchan / 8 * m_header.m_npol * m_header.m_nbits;
    // total number of downloding bytes to each channel:
    const long long QUantTotalChannelBytes = m_header.m_nblocksize * m_header.m_nbits / 8 / m_header.m_nchan;
    const long long QUantChunkComplexNumbers = lenChunk * m_header.m_nchan * m_header.m_npol / 2;
    FILE** prb_File = (FILE**)malloc(sizeof(FILE*));
    if (!prb_File) {
        // Handle memory allocation failure
        return 1;
    }

    openFileReadingStream(prb_File);

    
        
    for (int nB = 0; nB < outChunkHeader.m_numBlock +1; ++nB)
    {     

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus));
            // Handle the error appropriately
        }

        createCurrentTelescopeHeader(prb_File);


        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus));
            // Handle the error appropriately
        }
        // number of downloding bytes to each chunk:
        const long long QUantChunkBytes = lenChunk * m_header.m_nchan / 8 * m_header.m_npol * m_header.m_nbits;
        // total number of downloding bytes to each channel:
        const long long QUantTotalChannelBytes = m_header.m_nblocksize / m_header.m_nchan;
        // total number of downloading complex numbers of channel:
        const long long QUantChannelComplexNumbers = lenChunk * m_header.m_npol / 2;

        const int NumChunks = (m_header.m_nblocksize + QUantChunkBytes - 1) / QUantChunkBytes;

        // !6

        // 7. remains of not readed elements in block
        //long long iremainedBytes = m_header.m_nblocksize;
        float val_coherent_d;
        // !7
        float valSNR = -1;
        int argmaxRow = -1;
        int argmaxCol = -1;
        float coherentDedisp = -1.;

        for (int j = 0; j < NumChunks; ++j)
        {
        
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus));
                // Handle the error appropriately
            }
            download_and_unpack_chunk(prb_File, lenChunk, j, (inp_type_*)d_parrInput, pcmparrRawSignalCur);
            long long position4 = ftell(*prb_File);

            if (!((nB == outChunkHeader.m_numBlock) && (j == outChunkHeader.m_numChunk)))
            {
                continue;
            }

            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus));
                // Handle the error appropriately
            }

            /*std::vector<inp_type_ > data0(quantDownloadingBytes/4, 0);
            cudaMemcpy(data0.data(), d_parrInput, quantDownloadingBytes / 4 * sizeof(inp_type_ ),
                cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();*/

            /*std::vector <std::complex<float>> data4(lenChunk* m_header.m_nchan * m_header.m_npol/2, 0);
            cudaMemcpy(data4.data(), pcmparrRawSignalCur, lenChunk * m_header.m_nchan * m_header.m_npol / 2 * sizeof(cufftComplex),
                cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();*/
                    pChunk->detailedChunkProcessing(
                        outChunkHeader
                        , plan0
                        , plan1
                        , pcmparrRawSignalCur
                        , d_arrfdmt_norm
                        , pAuxBuff_fdmt
                        , pcarrTemp
                        , pcarrCD_Out
                        , pcarrBuff
                        , pInpOutBuffFdmt
                        , pFRg);

        }       
    }
    closeFileReadingStream(prb_File);
    free(prb_File);
    cudaFree(pcmparrRawSignalCur);
    cudaFree(d_arrfdmt_norm);
    cudaFree(pAuxBuff_fdmt);
    cudaFree(pcarrTemp); //2	
    cudaFree(pcarrCD_Out);//3
    cudaFree(pcarrBuff);//3
    cudaFree(pInpOutBuffFdmt);
    cudaFree(d_parrInput);
    delete pChunk;
    cufftDestroy(plan0);
    cufftDestroy(plan1);
    return 0;
 }
//    FILE* rb_File = fopen( m_strInpPath, "rb");
//    if (!navigateToBlock(rb_File, outChunkHeader.m_numBlock + 1))
//    {
//        return false;
//    }
//
//    // 2.allocation memory for parametrs in CPU
//    /*int nbits = 0;
//    float chanBW = 0;
//    int npol = 0;
//    bool bdirectIO = 0;
//    float centfreq = 0;
//    int nchan = 0;
//    float obsBW = 0;
//    long long nblocksize = 0;
//    EN_telescope TELESCOP = GBT;
//    int ireturn = 0;
//    float tresolution = 0.;*/
//    // !2
//
//    
//
//    if (m_header.m_nbits / 8 != sizeof(inp_type_))
//    {
//        std::cout << "check up Constants.h, inp_type_  " << std::endl;
//        return -1;
//    }
//
//   
//
//
//
//    cufftHandle plan0 = NULL;
//    cufftHandle plan1 = NULL;
//    int lenChunk = 0;
//    char* d_parrInput = NULL;
//    cufftComplex* pcmparrRawSignalCur = NULL;
//    CFdmtU fdmt;
//    void* pAuxBuff_fdmt = NULL;
//    fdmt_type_* d_arrfdmt_norm = NULL;
//    cufftComplex* pcarrTemp = NULL; //2	
//    cufftComplex* pcarrCD_Out = NULL;//3
//    cufftComplex* pcarrBuff = NULL;//3
//    char* pInpOutBuffFdmt = NULL;
//    CChunk* pChunk = new CChunk();
//
//
//
//    CChunk::preparations_and_memoryAllocations(m_header
//        , m_pulse_length
//        , m_d_max
//        , m_sigma_bound
//        , m_length_sum_wnd
//        , &lenChunk
//        , &plan0, &plan1, &fdmt, &d_parrInput, &pcmparrRawSignalCur
//        , &pAuxBuff_fdmt, &d_arrfdmt_norm, &pcarrTemp, &pcarrCD_Out
//        , &pcarrBuff, &pInpOutBuffFdmt, &pChunk);
//
//
//
//
//
//    // 4. memory allocation in GPU
//    // total number of downloding bytes to each chunk:
//   // number of downloding bytes to each chunk:
//    const long long QUantChunkBytes = lenChunk * m_header.m_nchan / 8 * m_header.m_npol * m_header.m_nbits;
//    // total number of downloding bytes to each channel:
//    const long long QUantTotalChannelBytes = m_header.m_nblocksize / m_header.m_nchan;
//    // total number of downloading complex numbers of channel:
//    const long long QUantChannelComplexNumbers = lenChunk * m_header.m_npol / 2;
//
//    
//    
//
//    const int NumChunks = (m_header.m_nblocksize + QUantChunkBytes - 1) / QUantChunkBytes;
//
//    for (int j = 0; j < outChunkHeader.m_numChunk +1; ++j)
//    {
//        long long quantDownloadingBytes = QUantChunkBytes;
//        //
//        if (j == (NumChunks - 1))
//        {
//            long long position = ftell(rb_File);
//            long long offset = -QUantChunkBytes / m_header.m_nchan * j + QUantTotalChannelBytes - QUantChunkBytes / m_header.m_nchan;
//            fseek(rb_File, offset, SEEK_CUR);
//            long long position1 = ftell(rb_File);
//            int yy = 0;
//
//        }
//
//
//
//        size_t sz = downloadChunk(rb_File, (char*)d_parrInput, quantDownloadingBytes);
//        if (j < outChunkHeader.m_numChunk)
//        {
//            continue;
//        }
//
//
//        
//
//       /* std::vector<inp_type_ > data0(quantDownloadingBytes / 4, 0);
//        cudaMemcpy(data0.data(), d_parrInput, quantDownloadingBytes / 4 * sizeof(inp_type_),
//            cudaMemcpyDeviceToHost);
//        cudaDeviceSynchronize();*/
//
//        const dim3 blockSize = dim3(1024, 1, 1);
//        const dim3 gridSize = dim3((lenChunk + blockSize.x - 1) / blockSize.x, m_header.m_nchan, 1);
//        unpackInput << < gridSize, blockSize >> > (pcmparrRawSignalCur, (inp_type_*)d_parrInput, lenChunk, m_header.m_nchan, m_header.m_npol);
//        cudaDeviceSynchronize();
//
//        /*std::vector <std::complex<float>> data4(lenChunk * m_header.m_nchan * m_header.m_npol / 2, 0);
//        cudaMemcpy(data4.data(), pcmparrRawSignalCur, lenChunk * m_header.m_nchan * m_header.m_npol / 2 * sizeof(cufftComplex),
//            cudaMemcpyDeviceToHost);
//        cudaDeviceSynchronize();*/
//
//
//        pChunk->detailedChunkProcessing(
//            outChunkHeader
//            , plan0
//            , plan1
//            , pcmparrRawSignalCur
//            , d_arrfdmt_norm
//            , pAuxBuff_fdmt
//            , pcarrTemp
//            , pcarrCD_Out
//            , pcarrBuff
//            , pInpOutBuffFdmt
//            , pFRg);
//
//        cudaFree(pcmparrRawSignalCur);
//        cudaFree(d_arrfdmt_norm);
//        cudaFree(pAuxBuff_fdmt);
//        cudaFree(pcarrTemp); //2	
//        cudaFree(pcarrCD_Out);//3
//        cudaFree(pcarrBuff);//3
//        cudaFree(pInpOutBuffFdmt);
//        cudaFree(d_parrInput);
//        delete pChunk;
//        cufftDestroy(plan0);
//        cufftDestroy(plan1);
//
//    }

  //  return true;
//}
//-----------------------------------------------------------------
void CSession_gpu::download_and_unpack_chunk(FILE** prb_File, const long long lenChunk, const int j
    , inp_type_* d_parrInput, cufftComplex* pcmparrRawSignalCur)
{

 }
//------------------------------------
bool CSession_gpu::navigateToBlock(FILE* rb_File,const int IBlockNum)
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


//-----------------------------------------------------------------
//INPUT:
// d_mtrxSig - input matrix, dimentions nchan x (lenChunk * npol) 
// lenChunk - number comlex elements in time series
// nchan - number of channels
// npol - number of polarizations, =2 || 4
// OUTPUT:
// pcmparrRawSignalCur - matrix with raw signal, corresponding with chunk

__global__
void unpackInput(cufftComplex* pcmparrRawSignalCur, inp_type_* d_parrInput, const int  lenChunk
    , const int  nchan, const int  npol)
{
    const int inChan = blockIdx.y;

    const int inBlockCur = blockIdx.x;

    unsigned int  numElemColOut = inBlockCur * blockDim.x + threadIdx.x;
    unsigned int  numElemColInp = npol * numElemColOut;
    if (numElemColOut >= lenChunk)
    {
        return;
    }
    for (int i = 0; i < npol / 2; ++i)
    {
        pcmparrRawSignalCur[(inChan * npol / 2 + i) * lenChunk + numElemColOut].x
            =  (float)d_parrInput[inChan * lenChunk * npol + numElemColInp + 2 * i];
        pcmparrRawSignalCur[(inChan * npol / 2 + i) * lenChunk + numElemColOut].y
            =  (float)d_parrInput[inChan * lenChunk * npol + numElemColInp + 2 * i + 1];
    }
}
//-----------------------------------------------------------------
bool CSession_gpu::do_plan_and_memAlloc( int* pLenChunk
    , cufftHandle* pplan0, cufftHandle* pplan1, CFdmtU* pfdmt, char** d_pparrInput, cufftComplex** ppcmparrRawSignalCur
    , void** ppAuxBuff_fdmt, fdmt_type_** d_parrfdmt_norm
    , cufftComplex** ppcarrTemp
    , cufftComplex** ppcarrCD_Out
    , cufftComplex** ppcarrBuff, char** ppInpOutBuffFdmt, CChunk** ppChunk)

{
    // 2.allocation memory for parametrs
  //  in CPU
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
    // !2
    FILE* rb_File = fopen(m_strInpPath, "rb");
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
        return false;
    }
    fclose(rb_File);
    if (nbits / 8 != sizeof(inp_type_))
    {
        std::cout << "check up Constants.h, inp_type_  " << std::endl;
        return -1;
    }

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
   

    plan_and_memAlloc( pLenChunk
        , pplan0, pplan1, pfdmt, d_pparrInput, ppcmparrRawSignalCur
        , ppAuxBuff_fdmt, d_parrfdmt_norm, ppcarrTemp, ppcarrCD_Out
        , ppcarrBuff, ppInpOutBuffFdmt, ppChunk);
}

//------------------------------------------------------------------------------

void CSession_gpu::plan_and_memAlloc(
    int* pLenChunk
    , cufftHandle* pplan0, cufftHandle* pplan1, CFdmtU* pfdmt, char** d_pparrInput, cufftComplex** ppcmparrRawSignalCur
    , void** ppAuxBuff_fdmt, fdmt_type_** d_parrfdmt_norm
    , cufftComplex** ppcarrTemp
    , cufftComplex** ppcarrCD_Out
    , cufftComplex** ppcarrBuff, char** ppInpOutBuffFdmt, CChunk** ppChunk)
{
    cudaError_t cudaStatus;
    const float VAlFmin =  m_header.m_centfreq - ((float) m_header.m_nchan) *  m_header.m_chanBW / 2.0;
    const float VAlFmax =  m_header.m_centfreq + ((float) m_header.m_nchan) *  m_header.m_chanBW / 2.0;
    // 3.2 calculate standard len_sft and LenChunk    
    const int len_sft = calc_len_sft(fabs( m_header.m_chanBW), m_pulse_length);
    *pLenChunk = _calcLenChunk_( m_header, len_sft, m_pulse_length, m_d_max);


    // 3.3 cuFFT plans preparations

    cufftCreate(pplan0);
    checkCudaErrors(cufftPlan1d(pplan0, *pLenChunk, CUFFT_C2C,  m_header.m_nchan *  m_header.m_npol / 2));



    cufftCreate(pplan1);
    checkCudaErrors(cufftPlan1d(pplan1, len_sft, CUFFT_C2C, (*pLenChunk) *  m_header.m_nchan *  m_header.m_npol / 2 / len_sft));



    // !3

    // 4. memory allocation in GPU
    // total number of downloding bytes to each file:
    const long long QUantDownloadingBytesForChunk = (*pLenChunk) *  m_header.m_nchan / 8 *  m_header.m_nbits *  m_header.m_npol;

    const long long QUantBlockComplexNumbers = (*pLenChunk) *  m_header.m_nchan *  m_header.m_npol / 2;



    checkCudaErrors(cudaMallocManaged((void**)d_pparrInput, QUantDownloadingBytesForChunk * sizeof(char)));


    checkCudaErrors(cudaMalloc((void**)ppcmparrRawSignalCur, QUantBlockComplexNumbers * sizeof(cufftComplex)));
    // 2!



    // 4.memory allocation for auxillary buffer for fdmt   
       // there is  quantity of real channels
    const int NChan_fdmt_act = len_sft *  m_header.m_nchan;
    (*pfdmt) = CFdmtU(
        VAlFmin
        , VAlFmax
        , NChan_fdmt_act
        , (*pLenChunk) / len_sft
        , m_pulse_length
        , m_d_max
        , len_sft);



    size_t szBuff_fdmt = pfdmt->calcSizeAuxBuff_fdmt_();

    checkCudaErrors(cudaMalloc(ppAuxBuff_fdmt, szBuff_fdmt));
    // 4!


    // 3. memory allocation for fdmt_ones on GPU  ????
    size_t szBuff_fdmt_output = pfdmt->calc_size_output();

    checkCudaErrors(cudaMalloc((void**)d_parrfdmt_norm, szBuff_fdmt_output));
    //// 6. calculation fdmt ones
    pfdmt->process_image(nullptr      // on-device input image
        , *ppAuxBuff_fdmt
        , *d_parrfdmt_norm	// OUTPUT image,
        , true);

    // 3!




    // 5. memory allocation for the 3 auxillary cufftComplex  arrays on GPU	
    //cufftComplex* pffted_rowsignal = NULL; //1	



    checkCudaErrors(cudaMalloc((void**)ppcarrTemp, QUantBlockComplexNumbers * sizeof(cufftComplex)));

    checkCudaErrors(cudaMalloc((void**)ppcarrCD_Out, QUantBlockComplexNumbers * sizeof(cufftComplex)));

    checkCudaErrors(cudaMalloc((void**)ppcarrBuff, QUantBlockComplexNumbers * sizeof(cufftComplex)));
    // !5

    // 5. memory allocation for the 2 auxillary arrays on GPU for input and output of FDMT	
    size_t szInpOut_fdmt = pfdmt->calc_size_output() + pfdmt->calc_size_input();

    checkCudaErrors(cudaMalloc((void**)ppInpOutBuffFdmt, szInpOut_fdmt));

    // 5!

    // !4	
    **ppChunk = CChunk(
        VAlFmin
        , VAlFmax
        ,  m_header.m_npol
        ,  m_header.m_nchan
        , (*pLenChunk)
        , len_sft
        , 0
        , 0
        ,  m_header.m_nbits
        , m_d_max
        , m_sigma_bound
        , m_length_sum_wnd
        , *pfdmt
        , m_pulse_length
    );

}
//------------------------------------------------------------

long long CSession_gpu::_calcLenChunk_(CTelescopeHeader header, const int nsft
    , const float pulse_length, const float d_max)
{
    const int nchan_actual = nsft * header.m_nchan;

    long long len = 0;
    for (len = 1 << 9; len < 1 << 30; len <<= 1)
    {
        CFdmtU fdmt(
            header.m_centfreq - header.m_chanBW * header.m_nchan / 2.
            , header.m_centfreq + header.m_chanBW * header.m_nchan / 2.
            , nchan_actual
            , len
            , pulse_length
            , d_max
            , nsft
        );
        long long size0 = fdmt.calcSizeAuxBuff_fdmt_();
        long long size_fdmt_inp = fdmt.calc_size_input();
        long long size_fdmt_out = fdmt.calc_size_output();
        long long size_fdmt_norm = size_fdmt_out;
        long long irest = header.m_nchan * header.m_npol * header.m_nbits / 8 // input buff
            + header.m_nchan * header.m_npol / 2 * sizeof(cufftComplex)
            + 3 * header.m_nchan * header.m_npol * sizeof(cufftComplex) / 2
            + 2 * header.m_nchan * sizeof(float);
        irest *= len;

        long long rez = size0 + size_fdmt_inp + size_fdmt_out + size_fdmt_norm + irest;
        if (rez > 0.98 * TOtal_GPU_Bytes)
        {
            return len / 2;
        }

    }
    return -1;
}

