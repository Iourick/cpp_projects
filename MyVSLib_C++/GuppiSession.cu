#include "GuppiSession.cuh"
#include <string>
#include "stdio.h"
#include <iostream>
//#include "Block_m_v1.cuh"
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
 

CGuppiSession::~CGuppiSession()
{ 
    if (m_pvctSuccessHeaders)
    {
        delete m_pvctSuccessHeaders;
    }

}
//-------------------------------------------
CGuppiSession::CGuppiSession()
{    
    
    memset(m_strGuppiPath, 0, MAX_PATH_LENGTH * sizeof(char));
    memset(m_strOutPutPath, 0, MAX_PATH_LENGTH * sizeof(char));
    m_pvctSuccessHeaders = new std::vector<COutChunkHeader>();
    m_header = CTelescopeHeader();
    m_pulse_length = 1.0E-6;
    m_d_max = 0.;
    m_sigma_bound = 10.;
    m_length_sum_wnd = 10;
    
}

//--------------------------------------------
CGuppiSession::CGuppiSession(const  CGuppiSession& R)
{  
    memcpy(m_strGuppiPath, R.m_strGuppiPath, MAX_PATH_LENGTH * sizeof(char));
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
CGuppiSession& CGuppiSession::operator=(const CGuppiSession& R)
{
    if (this == &R)
    {
        return *this;
    }   
    
    memcpy(m_strGuppiPath, R.m_strGuppiPath, MAX_PATH_LENGTH * sizeof(char));
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
CGuppiSession::CGuppiSession(const char* strGuppiPath, const char* strOutPutPath, const float t_p
,const double d_max, const float sigma_bound, const int length_sum_wnd)
{
    strcpy(m_strOutPutPath, strOutPutPath);
    strcpy(m_strGuppiPath, strGuppiPath);   
    m_pvctSuccessHeaders = new std::vector<COutChunkHeader>();
    m_pulse_length = t_p;
    m_d_max = d_max;
    m_sigma_bound = sigma_bound;
    m_length_sum_wnd = length_sum_wnd;
}
//------------------------------------
int CGuppiSession::calcQuantRemainBlocks(FILE* rb_File,unsigned long long* pilength)
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
    fseek(rb_File, position, SEEK_SET);
    return ireturn;
}

//---------------------------------------------------------------
int CGuppiSession::launch()
{
    cudaError_t cudaStatus;
    FILE* rb_File = fopen(m_strGuppiPath, "rb");
    // calc quantity sessions
    unsigned long long ilength = -1;

    // 1. blocks quantity calculation
    unsigned long long position = ftell(rb_File);
    const int IBlock = calcQuantRemainBlocks(rb_File ,&ilength);
    
    fseek(rb_File, position, SEEK_SET);
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
    float tresolution = 0.;
    // !2

    // 3. cuFFT plans preparations
       // 3.1 reading first header
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
    )
        )
    {
        return -1;
    }

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
    fseek(rb_File, 0, SEEK_SET);



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
    

    
    CChunk::preparations_and_memoryAllocations(m_header
        , m_pulse_length
        , m_d_max
        , m_sigma_bound
        , m_length_sum_wnd
        , &lenChunk
        , &plan0, &plan1, &fdmt, &d_parrInput, &pcmparrRawSignalCur
        , &pAuxBuff_fdmt, &d_arrfdmt_norm, &pcarrTemp, &pcarrCD_Out
        , &pcarrBuff, &pInpOutBuffFdmt, &pChunk);

    

    

    // 4. memory allocation in GPU
    // total number of downloding bytes to each chunk:
    const long long QUantChunkBytes = lenChunk * m_header.m_nchan / 8 * m_header.m_npol * m_header.m_nbits;
    // total number of downloding bytes to each channel:
    const long long QUantTotalChannelBytes = nblocksize * m_header.m_nbits / 8 / m_header.m_nchan;
    const long long QUantChunkComplexNumbers = lenChunk * m_header.m_nchan * m_header.m_npol / 2;

    
   

    // 5!

    // !4

    // 3. Performing a loop using the variable nS, nS = 0,..,IBlock. 
    //IBlock - number of bulks
    
    for (int nB = 0; nB < IBlock; ++nB)        
    {   
        
        cout << "                               BLOCK=  " << nB <<endl;
        // 3.1. reading info from current bulk header
        // After return 
        // file cursor is installed on beginning of data block
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
        )
            )
        {            
            return -1;
        }        
        long long position1 = ftell(rb_File);
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

        //-----------------------------------------
        //------------------------------------------
        // number of downloding bytes to each chunk:
        const long long QUantChunkBytes = lenChunk * m_header.m_nchan / 8 * m_header.m_npol * m_header.m_nbits;
        // total number of downloding bytes to each channel:
        const long long QUantTotalChannelBytes = m_header.m_nblocksize  / m_header.m_nchan;
        // total number of downloading complex numbers of channel:
        const long long QUantChannelComplexNumbers = lenChunk * m_header.m_npol / 2;

        const int NumChunks = (m_header.m_nblocksize + QUantChunkBytes - 1) / QUantChunkBytes;


        
        // !6

        // 7. remains of not readed elements in block
        long long iremainedBytes = m_header.m_nblocksize;
        float val_coherent_d;
        // !7

        

       

        float valSNR = -1;
        int argmaxRow = -1;
        int argmaxCol = -1;
        float coherentDedisp = -1.;

        

        
        for (int j = 0; j < NumChunks; ++j)
        {
            long long position2 = ftell(rb_File);
            long long quantDownloadingBytes = QUantChunkBytes;
            //
            if (j == (NumChunks - 1))
            {
                long long position = ftell(rb_File);
                long long offset =  - QUantChunkBytes / m_header.m_nchan * j + QUantTotalChannelBytes - QUantChunkBytes / m_header.m_nchan;
                fseek(rb_File, offset, SEEK_CUR);
                long long position1 = ftell(rb_File);
                int yy = 0;

            }          
           
            long long position3 = ftell(rb_File);

            size_t sz = downloadChunk(rb_File, (char*)d_parrInput, quantDownloadingBytes);


            long long position4 = ftell(rb_File);

            iremainedBytes = iremainedBytes - quantDownloadingBytes;

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
        // rewind to the beginning of the data block
        fseek(rb_File, -QUantTotalChannelBytes, SEEK_CUR);
        

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

        
        
        //------------------------------------------------------------
        //-------------------------------------------------------------


        

        

       
        
        unsigned long long ioffset = m_header.m_nblocksize;
        
        if (bdirectIO)
        {
            unsigned long long num = (ioffset + 511) / 512;
            ioffset = num * 512;
        }

        fseek(rb_File, ioffset, SEEK_CUR);  
    }
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
//---------------------------------------------------------
bool CGuppiSession::analyzeChunk(const COutChunkHeader outChunkHeader, CFragment* pFRg)
{
    FILE* rb_File = fopen(m_strGuppiPath, "rb");
    if (!navigateToBlock(rb_File, outChunkHeader.m_numBlock + 1))
    {
        return false;
    }

    // 2.allocation memory for parametrs in CPU
    /*int nbits = 0;
    float chanBW = 0;
    int npol = 0;
    bool bdirectIO = 0;
    float centfreq = 0;
    int nchan = 0;
    float obsBW = 0;
    long long nblocksize = 0;
    EN_telescope TELESCOP = GBT;
    int ireturn = 0;
    float tresolution = 0.;*/
    // !2

    

    if (m_header.m_nbits / 8 != sizeof(inp_type_))
    {
        std::cout << "check up Constants.h, inp_type_  " << std::endl;
        return -1;
    }

   



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



    CChunk::preparations_and_memoryAllocations(m_header
        , m_pulse_length
        , m_d_max
        , m_sigma_bound
        , m_length_sum_wnd
        , &lenChunk
        , &plan0, &plan1, &fdmt, &d_parrInput, &pcmparrRawSignalCur
        , &pAuxBuff_fdmt, &d_arrfdmt_norm, &pcarrTemp, &pcarrCD_Out
        , &pcarrBuff, &pInpOutBuffFdmt, &pChunk);





    // 4. memory allocation in GPU
    // total number of downloding bytes to each chunk:
   // number of downloding bytes to each chunk:
    const long long QUantChunkBytes = lenChunk * m_header.m_nchan / 8 * m_header.m_npol * m_header.m_nbits;
    // total number of downloding bytes to each channel:
    const long long QUantTotalChannelBytes = m_header.m_nblocksize / m_header.m_nchan;
    // total number of downloading complex numbers of channel:
    const long long QUantChannelComplexNumbers = lenChunk * m_header.m_npol / 2;

    
    

    const int NumChunks = (m_header.m_nblocksize + QUantChunkBytes - 1) / QUantChunkBytes;

    for (int j = 0; j < outChunkHeader.m_numChunk +1; ++j)
    {
        long long quantDownloadingBytes = QUantChunkBytes;
        //
        if (j == (NumChunks - 1))
        {
            long long position = ftell(rb_File);
            long long offset = -QUantChunkBytes / m_header.m_nchan * j + QUantTotalChannelBytes - QUantChunkBytes / m_header.m_nchan;
            fseek(rb_File, offset, SEEK_CUR);
            long long position1 = ftell(rb_File);
            int yy = 0;

        }



        size_t sz = downloadChunk(rb_File, (char*)d_parrInput, quantDownloadingBytes);
        if (j < outChunkHeader.m_numChunk)
        {
            continue;
        }


        

       /* std::vector<inp_type_ > data0(quantDownloadingBytes / 4, 0);
        cudaMemcpy(data0.data(), d_parrInput, quantDownloadingBytes / 4 * sizeof(inp_type_),
            cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();*/

        const dim3 blockSize = dim3(1024, 1, 1);
        const dim3 gridSize = dim3((lenChunk + blockSize.x - 1) / blockSize.x, m_header.m_nchan, 1);
        unpackInput << < gridSize, blockSize >> > (pcmparrRawSignalCur, (inp_type_*)d_parrInput, lenChunk, m_header.m_nchan, m_header.m_npol);
        cudaDeviceSynchronize();

        /*std::vector <std::complex<float>> data4(lenChunk * m_header.m_nchan * m_header.m_npol / 2, 0);
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

    }
//
//    // 3.2 calculate standard len_sft and LenChunk    
//    const int len_sft = calc_len_sft(fabs(m_header.m_chanBW), m_pulse_length);
//    const unsigned int LenChunk = calcLenChunk(len_sft);   
//    const bool bCHannel_order = (m_header.m_chanBW > 0.) ? true : false;
//    //
//    // 3.3 cuFFT plans preparations
//    cufftHandle plan0 = NULL;
//    cufftCreate(&plan0);
//    checkCudaErrors(cufftPlan1d(&plan0, LenChunk, CUFFT_C2C, m_header.m_nchan * m_header.m_npol / 2));
//
//    cufftHandle plan1 = NULL;
//    cufftCreate(&plan1);
//    checkCudaErrors(cufftPlan1d(&plan1, len_sft, CUFFT_C2C, LenChunk * m_header.m_nchan * m_header.m_npol / 2 / len_sft));
//
//    cufftHandle* pcuPlan0 = &plan0;
//    cufftHandle* pcuPlan1 = &plan1;
//    // !3
//
//    // 4. memory allocation in GPU
//    // total number of downloding bytes to each chunk:
//    const long long QUantTotalChunkBytes = LenChunk * m_header.m_nchan / 8 * m_header.m_npol * m_header.m_nbits;
//    // total number of downloding bytes to each channel:
//    const long long QUantTotalChannelBytes = m_header.m_nblocksize * m_header.m_nbits / 8 / m_header.m_nchan;
//    const long long QUantChunkComplexNumbers = LenChunk * m_header.m_nchan * m_header.m_npol / 2;
//
//
//    cufftComplex* pcmparrRawSignalCur = NULL;
//    checkCudaErrors(cudaMallocManaged((void**)&pcmparrRawSignalCur, QUantChunkComplexNumbers * sizeof(cufftComplex)));
//    // 2!
//
//     // 4.memory allocation for auxillary buffer for fdmt
//    const float VAlFmin = m_header.m_centfreq - ((float)m_header.m_nchan) * m_header.m_chanBW / 2.0;
//    const float VAlFmax = m_header.m_centfreq + ((float)m_header.m_nchan) * m_header.m_chanBW / 2.0;
//    // 3.2 calculate standard len_sft and LenChunk    
//
//    const int NChan_fdmt_act = len_sft * m_header.m_nchan;
//    CFdmtU* pfdmt = new CFdmtU(
//        VAlFmin
//        , VAlFmax
//        , NChan_fdmt_act
//        , LenChunk / len_sft
//        , m_pulse_length
//        , m_d_max
//        , len_sft);
//    size_t szBuff_fdmt = pfdmt->calcSizeAuxBuff_fdmt_();
//    void* pAuxBuff_fdmt = 0;
//    checkCudaErrors(cudaMalloc(&pAuxBuff_fdmt, szBuff_fdmt));
//    // 4!
//
//
//
//    // 3. memory allocation for fdmt_ones on GPU  ????
//    size_t szBuff_fdmt_output = pfdmt->calc_size_output();
//    fdmt_type_* d_arrfdmt_norm = 0;
//    checkCudaErrors(cudaMalloc((void**)&d_arrfdmt_norm, szBuff_fdmt_output * sizeof(fdmt_type_)));
//    // 6. calculation fdmt ones
//    pfdmt->process_image(nullptr      // on-device input image
//        , pAuxBuff_fdmt
//        , d_arrfdmt_norm	// OUTPUT image, dim = IDeltaT x IImgcols
//        , true);
//
//    // 3!
//
//    // 5. memory allocation for the 3 auxillary cufftComplex  arrays on GPU	
//    //cufftComplex* pffted_rowsignal = NULL; //1	
//    cufftComplex* pcarrTemp = NULL; //2	
//    cufftComplex* pcarrCD_Out = NULL;//3
//    cufftComplex* pcarrBuff = NULL;//3
//
//
//    checkCudaErrors(cudaMallocManaged((void**)&pcarrTemp, QUantChunkComplexNumbers * sizeof(cufftComplex)));
//
//    checkCudaErrors(cudaMalloc((void**)&pcarrCD_Out, QUantChunkComplexNumbers * sizeof(cufftComplex)));
//
//    checkCudaErrors(cudaMalloc((void**)&pcarrBuff, QUantChunkComplexNumbers * sizeof(cufftComplex)));
//    // !5
//
//    // 5. memory allocation for the 2 auxillary float  arrays on GPU	
//    float* pAuxBuff_flt = NULL;
//    checkCudaErrors(cudaMalloc((void**)&pAuxBuff_flt, 2 * LenChunk * m_header.m_nchan * sizeof(float)));
//
//    
//
//
//
////-------------------------------------------------------------------------------------
//    
//    CBlock* pBlock = new CBlock(
//        m_header.m_centfreq - fabs(m_header.m_chanBW) * m_header.m_nchan / 2.
//        , m_header.m_centfreq + fabs(m_header.m_chanBW) * m_header.m_nchan / 2.
//        , m_header.m_npol
//        , m_header.m_nblocksize
//        , m_header.m_nchan
//        , LenChunk
//        , len_sft
//        , outChunkHeader.m_numBlock
//        , m_header.m_nbits
//        , bCHannel_order
//        , m_d_max
//        , m_sigma_bound
//        , m_length_sum_wnd
//
//    );
//    
//
//    if (!pBlock->detailedChunkProcessing(rb_File
//        , outChunkHeader
//        , pcuPlan0
//        , pcuPlan1
//        , pcmparrRawSignalCur
//        , d_arrfdmt_norm
//        , pAuxBuff_fdmt
//        , pcarrTemp
//        , pcarrCD_Out
//        , pcarrBuff
//        , pAuxBuff_flt
//        , pFRg))
//    {
//        delete pBlock;
//        cudaFree(pcmparrRawSignalCur);
//        cudaFree(d_arrfdmt_norm);
//        cudaFree(pAuxBuff_fdmt);
//        cudaFree(pcarrTemp); //2	
//        cudaFree(pcarrCD_Out);//3
//        cudaFree(pcarrBuff);//3
//        cudaFree(pAuxBuff_flt);
//        cufftDestroy(plan0);
//        cufftDestroy(plan1);
//        return false;
//    }
//    delete pBlock;
//
//    cudaFree(pcmparrRawSignalCur);
//    cudaFree(d_arrfdmt_norm);
//    cudaFree(pAuxBuff_fdmt);
//    cudaFree(pcarrTemp); //2	
//    cudaFree(pcarrCD_Out);//3
//    cudaFree(pcarrBuff);//3
//    cudaFree(pAuxBuff_flt);
//
//    cufftDestroy(plan0);
//    cufftDestroy(plan1);
    return true;
}

//------------------------------------
bool CGuppiSession::navigateToBlock(FILE* rb_File,const int IBlockNum)
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
//------------------------------------------
size_t  CGuppiSession::downloadChunk(FILE* rb_File, char* d_parrInput, const long long QUantDownloadingBytes)
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
    fseek(rb_File,-(m_header.m_nchan -1) * quantTotalBytesPerChannel, SEEK_CUR);
    long long position5 = ftell(rb_File);
    return sz_rez;
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

