#include "Session_m.cuh"
#include <string>
#include "stdio.h"
#include <iostream>
#include "Block_m.cuh"
#include "OutChunk.h"
#include <cufft.h>
#include "fdmtU_cu.cuh"
#include <stdlib.h>
#include "Fragment.cuh"
#include "helper_functions.h"
#include "helper_cuda.h"
 

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
    float tresolution = 0.;
    *pilength = 0;
    for (int i = 0; i < 1 << 26; ++i)
    {        
        long long pos0 = ftell(m_rbFile);
        if (!CTelescopeHeader::readGuppiHeader(
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
    float tresolution = 0.;
    // !2

    // 3. cuFFT plans preparations
       // 3.1 reading first header
    if (!CTelescopeHeader::readGuppiHeader(
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
    fseek(m_rbFile, 0, SEEK_SET);
    // 3.2 calculate standard len_sft and LenChunk    
    const int len_sft = calc_len_sft(fabs(chanBW), m_t_p);    
    const unsigned int LenChunk = calcLenChunk(len_sft);
    //
    // 3.3 cuFFT plans preparations
    cufftHandle plan0 = NULL;
    cufftCreate(&plan0);
    checkCudaErrors(cufftPlan1d(&plan0, LenChunk, CUFFT_C2C, nchan * npol / 2));

    cufftHandle plan1 = NULL;
    cufftCreate(&plan1);
    checkCudaErrors(cufftPlan1d(&plan1, len_sft, CUFFT_C2C, LenChunk * nchan * npol / 2 / len_sft));

    cufftHandle* pcuPlan0 = &plan0;
    cufftHandle* pcuPlan1 = &plan1;
    // !3

    // 4. memory allocation in GPU
    // total number of downloding bytes to each chunk:
    const long long QUantTotalChunkBytes = LenChunk * nchan / 8 * npol * nbits;
    // total number of downloding bytes to each channel:
   // const long long QUantTotalChannelBytes = nblocksize * nbits / 8 / nchan;
    const long long QUantChunkComplexNumbers = LenChunk * nchan * npol / 2;

    
    char* parrInput = NULL;
    checkCudaErrors(cudaMallocManaged((void**)&parrInput, QUantTotalChunkBytes * sizeof(char)));

    cufftComplex* pcmparrRawSignalCur = NULL;
    checkCudaErrors(cudaMalloc((void**)&pcmparrRawSignalCur, QUantChunkComplexNumbers * sizeof(cufftComplex)));
    // 2!

    // 4.memory allocation for auxillary buffer for fdmt
    const int N_p = len_sft * nchan;
    unsigned int IMaxDT = calc_MaxDT(centfreq - fabs(obsBW) / 2.0, centfreq + fabs(obsBW) / 2.0
        , m_t_p, m_d_max);
    const int  IDeltaT = calc_IDeltaT(N_p, centfreq - fabs(obsBW) / 2.0, centfreq + fabs(obsBW) / 2.0, IMaxDT);
    size_t szBuff_fdmt = calcSizeAuxBuff_fdmt(N_p, LenChunk / len_sft
        , centfreq - fabs(obsBW) / 2.0, centfreq + fabs(obsBW) / 2.0, IMaxDT);
    void* pAuxBuff_fdmt = 0;
    checkCudaErrors(cudaMalloc(&pAuxBuff_fdmt, szBuff_fdmt));
    // 4!

    // 3. memory allocation for fdmt_ones on GPU  ????
    fdmt_type_* d_arrfdmt_norm = 0;
    checkCudaErrors(cudaMalloc((void**)&d_arrfdmt_norm, LenChunk * nchan * sizeof(fdmt_type_)));
    // 6. calculation fdmt ones
    fncFdmtU_cu(
        nullptr      // on-device input image
        , pAuxBuff_fdmt
        , N_p
        , LenChunk / len_sft // dimensions of input image 	
        , IDeltaT
        , centfreq - fabs(obsBW) / 2.0
        , centfreq + fabs(obsBW) / 2.0
        , IMaxDT
        , d_arrfdmt_norm	// OUTPUT image, dim = IDeltaT x IImgcols
        , true
    );
    // 3!

    

    // 5. memory allocation for the 3 auxillary cufftComplex  arrays on GPU	
    //cufftComplex* pffted_rowsignal = NULL; //1	
    cufftComplex* pcarrTemp = NULL; //2	
    cufftComplex* pcarrCD_Out = NULL;//3
    cufftComplex* pcarrBuff = NULL;//3








    checkCudaErrors(cudaMalloc((void**)&pcarrTemp, QUantChunkComplexNumbers * sizeof(cufftComplex)));

    checkCudaErrors(cudaMalloc((void**)&pcarrCD_Out, QUantChunkComplexNumbers * sizeof(cufftComplex)));

    checkCudaErrors(cudaMalloc((void**)&pcarrBuff, QUantChunkComplexNumbers * sizeof(cufftComplex)));
    // !5

    // 5. memory allocation for the 2 auxillary float  arrays on GPU	
    float* pAuxBuff_flt = NULL;
    checkCudaErrors(cudaMalloc((void**)&pAuxBuff_flt, 2 * LenChunk * nchan * sizeof(float)));

    // 5!

    // !4

    // 3. Performing a loop using the variable nS, nS = 0,..,IBlock. 
    //IBlock - number of bulks
    
    for (int nS = 0; nS < IBlock; ++nS)
    {   
        
        cout << "                               BLOCK=  " << nS <<endl;
        // 3.1. reading info from current bulk header
        // After return 
        // file cursor is installed on beginning of data block
        if (!CTelescopeHeader::readGuppiHeader(
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
            , &tresolution
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
            , tresolution
        );
        // 2!

        // calculate N_p
        const int len_sft = calc_len_sft(fabs(m_header.m_chanBW), m_t_p);

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

        pBlock->process(m_rbFile
            , pcuPlan0
            , pcuPlan1
            , parrInput
            , pcmparrRawSignalCur
            , d_arrfdmt_norm
            , pAuxBuff_fdmt
            , pcarrTemp	
            , pcarrCD_Out 
            , pcarrBuff
            , pAuxBuff_flt
            , m_pvctSuccessHeaders);
                
        delete pBlock;
        
        unsigned long long ioffset = m_header.m_nblocksize;
        
        if (bdirectIO)
        {
            unsigned long long num = (ioffset + 511) / 512;
            ioffset = num * 512;
        }

        fseek(m_rbFile, ioffset, SEEK_CUR);  
    }
    cudaFree(pcmparrRawSignalCur);
    cudaFree(d_arrfdmt_norm);
    cudaFree(pAuxBuff_fdmt);
    cudaFree(pcarrTemp); //2	
    cudaFree(pcarrCD_Out);//3
    cudaFree(pcarrBuff);//3
    cudaFree(pAuxBuff_flt);
    cudaFree(parrInput);

    cufftDestroy(plan0);
    cufftDestroy(plan1);
    return 0;
}
//---------------------------------------------------------
bool CSession::analyzeChunk(const COutChunkHeader outChunkHeader, CFragment* pFRg)
{
    if (!navigateToBlock(outChunkHeader.m_numBlock + 1))
    {
        return false;
    }

    // 3.2 calculate standard len_sft and LenChunk    
    const int len_sft = calc_len_sft(fabs(m_header.m_chanBW), m_t_p);
    const unsigned int LenChunk = calcLenChunk(len_sft);   
    const bool bCHannel_order = (m_header.m_chanBW > 0.) ? true : false;
    //
    // 3.3 cuFFT plans preparations
    cufftHandle plan0 = NULL;
    cufftCreate(&plan0);
    checkCudaErrors(cufftPlan1d(&plan0, LenChunk, CUFFT_C2C, m_header.m_nchan * m_header.m_npol / 2));

    cufftHandle plan1 = NULL;
    cufftCreate(&plan1);
    checkCudaErrors(cufftPlan1d(&plan1, len_sft, CUFFT_C2C, LenChunk * m_header.m_nchan * m_header.m_npol / 2 / len_sft));

    cufftHandle* pcuPlan0 = &plan0;
    cufftHandle* pcuPlan1 = &plan1;
    // !3

    // 4. memory allocation in GPU
    // total number of downloding bytes to each chunk:
    const long long QUantTotalChunkBytes = LenChunk * m_header.m_nchan / 8 * m_header.m_npol * m_header.m_nbits;
    // total number of downloding bytes to each channel:
    const long long QUantTotalChannelBytes = m_header.m_nblocksize * m_header.m_nbits / 8 / m_header.m_nchan;
    const long long QUantChunkComplexNumbers = LenChunk * m_header.m_nchan * m_header.m_npol / 2;


    cufftComplex* pcmparrRawSignalCur = NULL;
    checkCudaErrors(cudaMallocManaged((void**)&pcmparrRawSignalCur, QUantChunkComplexNumbers * sizeof(cufftComplex)));
    // 2!

    // 4.memory allocation for auxillary buffer for fdmt
    const int N_p = len_sft * m_header.m_nchan;
    unsigned int IMaxDT = len_sft * m_header.m_nchan;
    const int  IDeltaT = calc_IDeltaT(N_p, m_header.m_centfreq - fabs(m_header.m_obsBW) / 2.0, m_header.m_centfreq + fabs(m_header.m_obsBW) / 2.0, IMaxDT);
    size_t szBuff_fdmt = calcSizeAuxBuff_fdmt(N_p, LenChunk / len_sft
        , m_header.m_centfreq - fabs(m_header.m_obsBW) / 2.0, m_header.m_centfreq + fabs(m_header.m_obsBW) / 2.0, IMaxDT);
    void* pAuxBuff_fdmt = 0;
    checkCudaErrors(cudaMalloc(&pAuxBuff_fdmt, szBuff_fdmt));
    // 4!

    // 3. memory allocation for fdmt_ones on GPU  ????
    fdmt_type_* d_arrfdmt_norm = 0;
    checkCudaErrors(cudaMalloc((void**)&d_arrfdmt_norm, LenChunk * m_header.m_nchan * sizeof(fdmt_type_)));
    // 6. calculation fdmt ones
    fncFdmtU_cu(
        nullptr      // on-device input image
        , pAuxBuff_fdmt
        , N_p
        , LenChunk / len_sft // dimensions of input image 	
        , IDeltaT
        , m_header.m_centfreq - fabs(m_header.m_obsBW) / 2.0
        , m_header.m_centfreq + fabs(m_header.m_obsBW) / 2.0
        , IMaxDT
        , d_arrfdmt_norm	// OUTPUT image, dim = IDeltaT x IImgcols
        , true
    );
    // 3!



    // 5. memory allocation for the 3 auxillary cufftComplex  arrays on GPU	
    //cufftComplex* pffted_rowsignal = NULL; //1	
    cufftComplex* pcarrTemp = NULL; //2	
    cufftComplex* pcarrCD_Out = NULL;//3
    cufftComplex* pcarrBuff = NULL;//3


    checkCudaErrors(cudaMallocManaged((void**)&pcarrTemp, QUantChunkComplexNumbers * sizeof(cufftComplex)));

    checkCudaErrors(cudaMalloc((void**)&pcarrCD_Out, QUantChunkComplexNumbers * sizeof(cufftComplex)));

    checkCudaErrors(cudaMalloc((void**)&pcarrBuff, QUantChunkComplexNumbers * sizeof(cufftComplex)));
    // !5

    // 5. memory allocation for the 2 auxillary float  arrays on GPU	
    float* pAuxBuff_flt = NULL;
    checkCudaErrors(cudaMalloc((void**)&pAuxBuff_flt, 2 * LenChunk * m_header.m_nchan * sizeof(float)));

    



//-------------------------------------------------------------------------------------
    
    CBlock* pBlock = new CBlock(
        m_header.m_centfreq - fabs(m_header.m_chanBW) * m_header.m_nchan / 2.
        , m_header.m_centfreq + fabs(m_header.m_chanBW) * m_header.m_nchan / 2.
        , m_header.m_npol
        , m_header.m_nblocksize
        , m_header.m_nchan
        , LenChunk
        , len_sft
        , outChunkHeader.m_numBlock
        , m_header.m_nbits
        , bCHannel_order
        , m_d_max
        , m_sigma_bound
        , m_length_sum_wnd

    );
    

    if (!pBlock->detailedChunkProcessing(m_rbFile
        , outChunkHeader
        , pcuPlan0
        , pcuPlan1
        , pcmparrRawSignalCur
        , d_arrfdmt_norm
        , pAuxBuff_fdmt
        , pcarrTemp
        , pcarrCD_Out
        , pcarrBuff
        , pAuxBuff_flt
        , pFRg))
    {
        delete pBlock;
        cudaFree(pcmparrRawSignalCur);
        cudaFree(d_arrfdmt_norm);
        cudaFree(pAuxBuff_fdmt);
        cudaFree(pcarrTemp); //2	
        cudaFree(pcarrCD_Out);//3
        cudaFree(pcarrBuff);//3
        cudaFree(pAuxBuff_flt);
        cufftDestroy(plan0);
        cufftDestroy(plan1);
        return false;
    }
    delete pBlock;

    cudaFree(pcmparrRawSignalCur);
    cudaFree(d_arrfdmt_norm);
    cudaFree(pAuxBuff_fdmt);
    cudaFree(pcarrTemp); //2	
    cudaFree(pcarrCD_Out);//3
    cudaFree(pcarrBuff);//3
    cudaFree(pAuxBuff_flt);

    cufftDestroy(plan0);
    cufftDestroy(plan1);
    return true;
}
//-----------------------------------------------------------------
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

    float valDenominator = m_header.m_nchan * m_header.m_npol * m_header.m_nbits / 8 // input buff
        + m_header.m_nchan * m_header.m_npol / 2 * sizeof(cufftComplex)        
        + 2 * (IDeltaT + 1) * m_header.m_nchan * sizeof(fdmt_type_)    // fdmt buff
        + m_header.m_nchan * sizeof(fdmt_type_)    // fdmt normalization
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
    
    
    for (int i = 0; i < m_pvctSuccessHeaders->size(); ++i)
    {
        if (24 == i)
        {
            int hghg = 0;
        }
        char arrch[2000] = { 0 };
        char charrTemp[200] = { 0 };
        
        (*m_pvctSuccessHeaders)[i].createOutStr(charrTemp);
        strcat(arrch, charrTemp);
        memset(charrTemp, 0, 200 * sizeof(char));
        sprintf(charrTemp, ", Length of pulse= %.10e", m_t_p);
        strcat(arrch, charrTemp);
        //strcat(arrch, "\n");
        strcat(arrch, "\n");
        size_t elements_written = fwrite(arrch, sizeof(char), strlen(arrch), m_wb_file);
            
    }
    //size_t elements_written = fwrite(arrch, sizeof(char), strlen(arrch), m_wb_file);
}


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
    float tresolution = 0.;
    
    for (int i = 0; i < IBlockNum; ++i)
    {
        long long pos0 = ftell(m_rbFile);
        if (!CTelescopeHeader::readGuppiHeader(
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
            , &tresolution
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

        fseek(m_rbFile, ioffset, SEEK_CUR);

    }
    
    return true;
}

