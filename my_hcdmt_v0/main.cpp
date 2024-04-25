#include "main.h"
#include <iostream>

//#include <fftw3.h>
#include <GL/glut.h>
#include <fstream>
//#include "HybridC_v0.h"
//#include "utilites.h"
#include "OutChunk.h"
//#include "Fragment.h"


#include <array>
#include <string>
#include <vector>
#include <cstdlib> // For random value generation
#include <ctime>   // For seeding the random number generator
#include "npy.hpp"
#include <algorithm> 
//#include "read_and_write_log.h"
#include <chrono>
//#include "fileInput.h"
#include "DrawImg.h"
#include "Constants.h"
#include "SessionB.h"
#include "Fragment.h"
#include "Session_lofar_cpu.h"
#include "Session_lofar_gpu.cuh"

#define _CRT_SECURE_NO_WARNINGS
using namespace std;
using namespace std;



/************** DATA FOR LOFAR ****************************/
char PathInpFile[] = "D://BASSA//hdf5_data//L2012176_SAP000_B000_S0_P001_bf.h5";
char PathOutFold[] = "OutPutFold";
TYPE_OF_INP_FORMAT INP_FORMAT = FLOFAR;
TYPE_OF_PROCESSOR PROCESSOR = GPU;

char* pPathInpFile = PathInpFile;
char* pPathOutFold = PathOutFold;
double valD_min = 30.0;
double valD_max = 50.0;
double length_of_pulse = 5.12E-6 * 32.0;//;
float sigma_Bound = 12.;
int lenWindow = 1;
int nbin = 262144;
int nfft = 1;
/*************** ! DATA FOR LOFAR *****************************/


/***************   GUPPI ********************************************/
 //char PathInpFile[] = "D://weizmann//RAW_DATA//blc20_guppi_57991_49905_DIAG_FRB121102_0011.0007.raw";
 //char PathOutFold[] = "OutPutFold";
 //char* pPathInpFile = PathInpFile;
 //char* pPathOutFold = PathOutFold;
 //TYPE_OF_INP_FORMAT INP_FORMAT = GUPPI;
 //double valD_max = 96.73;
 //double length_of_pulse = 1.36E-6;
 //float sigma_Bound =12.;
 //int lenWindow = 10;
// int lenChunk = 32768;
/*************** !  GUPPI ********************************************/


/***************  TEST FOR GUPPI  nchan =1, npol = 2 ********************************************/
// char PathInpFile[] = "D://weizmann//RAW_DATA//rawImit_2pow20_nchan_1npol_2_float_.bin";//40.0E-8
// char PathOutFold[] = "OutPutFold";
//char* pPathInpFile = PathInpFile;
//char* pPathOutFold = PathOutFold;
//TYPE_OF_INP_FORMAT INP_FORMAT = GUPPI;
// double valD_max =  1.5;
// double length_of_pulse = 3.2E-7;
// float sigma_Bound = 100.0;
// int lenWindow = 10;
// int lenChunk = 32768;
/*************** ! TEST FOR GUPPI ********************************************/


     /***************  TEST FOR GUPPI  nchan =2, npol = 4 ********************************************/
     // char PathInpFile[] = "D://weizmann//RAW_DATA//rawImit_2pow20_nchan_2npol_4_float_.bin";//40.0E-8
     // char PathOutFold[] = "OutPutFold";
     //char* pPathInpFile = PathInpFile;
     //char* pPathOutFold = PathOutFold;
     //TYPE_OF_INP_FORMAT INP_FORMAT = GUPPI;
     // double valD_max =  1.5;
     // double length_of_pulse = 3.2E-7;
     // float sigma_Bound = 100.0;
     // int lenWindow = 10;
// int lenChunk = 32768;

     /*************** ! TEST FOR GUPPI ********************************************/

//
void showInputString(char* pPathLofarFile, char* pPathOutFold, float length_of_pulse
    , float VAlD_max, float sigma_Bound, int lenWindow, int nbins, int nfft)
{
    std::cout << "pPathLofarFile:    " << pPathLofarFile << std::endl;
    std::cout << "pPathOutFold:      " << pPathOutFold << std::endl;
    std::cout << "length_of_pulse =  " << length_of_pulse << std::endl;
    std::cout << "VAlD_max =         " << VAlD_max << std::endl;
    std::cout << "sigma_Bound =      " << sigma_Bound << std::endl;
    std::cout << "lenWindow =        " << lenWindow << std::endl;
    std::cout << "nbins =        " << nbins << std::endl;
    std::cout << "nfft =        " << nfft << std::endl;
}
//-------------------------------------------------------------------

bool launch_file_processing(TYPE_OF_PROCESSOR PROCESSOR, TYPE_OF_INP_FORMAT INP_FORMAT, char* pPathInpFile, char* pPathOutFold, const float length_of_pulse
    , const float valD_min, const float  valD_max, const float sigma_Bound, const int lenWindow, const int  nbin, const int  nfft
 , std::vector<std::vector<float>> *pvecImg,  int * pmsamp)
{
    CSessionB* pSession = nullptr;
   
    CSession_lofar_cpu* pSess_lofar_cpu = nullptr;
    CSession_lofar_gpu* pSess_lofar_gpu = nullptr;
    switch (INP_FORMAT)
    {
    case GUPPI:
        // pSess_guppi = new CSession_gpu_guppi(pPathInpFile, pPathOutFold, length_of_pulse, valD_max, sigma_Bound, lenWindow,lenChunk);
        // pSession = pSess_guppi;
        break;

    case FLOFAR:
        switch (PROCESSOR)
        {
        case CPU:
            pSess_lofar_cpu = new CSession_lofar_cpu(pPathInpFile, pPathOutFold, length_of_pulse
                , valD_min, valD_max, sigma_Bound, lenWindow, nbin, nfft);
            pSession = pSess_lofar_cpu;
            break;

        case GPU:
            pSess_lofar_gpu = new CSession_lofar_gpu(pPathInpFile, pPathOutFold, length_of_pulse
                , valD_min, valD_max, sigma_Bound, lenWindow, nbin, nfft);
            pSession = pSess_lofar_gpu;
            break;
        default: break;
        }       
        break;

    default:
        return -1;

    }
    
    if (-1 == pSession->launch(pvecImg, pmsamp))
    {
        pSession = nullptr;
        
        if (pSess_lofar_cpu)
        {
            delete   pSess_lofar_cpu;
        }

        if (pSess_lofar_gpu)
        {
            delete   pSess_lofar_gpu;
        }
        return -1;
    }

    if (pSession->m_pvctSuccessHeaders->size() > 0)
    {
        std::cout << "               Successful Chunk Numbers = " << pSession->m_pvctSuccessHeaders->size() << std::endl;
        //--------------------------------------

        char charrTemp[200] = { 0 };
        for (int i = 0; i < pSession->m_pvctSuccessHeaders->size(); ++i)
        {
            memset(charrTemp, 0, 200 * sizeof(char));
            (*(pSession->m_pvctSuccessHeaders))[i].createOutStr(charrTemp);
            std::cout << i + 1 << ". " << charrTemp << std::endl;
        }
    }
    else
    {
        std::cout << "               Successful Chunk Were Not Detected= " << std::endl;
        return 0;
    }

    char outputlogfile[300] = { 0 };
    strcpy(outputlogfile, "output.log");
    COutChunkHeader::writeReport(outputlogfile, pSession->m_pvctSuccessHeaders
        , length_of_pulse);

    pSession = nullptr;

}
int main(int argc, char** argv)
{
    /*int block = 105;
    int chunk = 15;
    int ov = 5;
    int num = (block - ov -1 ) / (chunk - ov) + 1;
    return;*/
    //defining by default input parameters of cmd line                                                              

// !

    if (argc > 1)
    {
        if (argc < 11)
        {
            std::cerr << "Usage: " << argv[0] << " -n <InpFile> -N <OutFold> -P <length_of_pulse> -b <tresh> -d <lenWin>" << std::endl;
            return 1;
        }
        for (int i = 1; i < argc; ++i)
        {
            if (std::string(argv[i]) == "-n")
            {
                pPathInpFile = argv[++i];
                continue;
            }
            if (std::string(argv[i]) == "-N")
            {
                pPathOutFold = argv[++i];
                continue;
            }
            if (std::string(argv[i]) == "-P")
            {
                length_of_pulse = std::atof(argv[++i]);
                continue;
            }
            if (std::string(argv[i]) == "-b")
            {
                sigma_Bound = std::atof(argv[++i]);
                continue;
            }

            if (std::string(argv[i]) == "-k")
            {
                valD_max = std::atof(argv[++i]);
                continue;
            }
            if (std::string(argv[i]) == "-d")
            {
                lenWindow = std::atoi(argv[++i]);
                continue;
            }
        }
    }
    showInputString(pPathInpFile, pPathOutFold, length_of_pulse, valD_max, sigma_Bound, lenWindow, nbin, nfft);


    CSessionB* pSession = nullptr;

    CSession_lofar_cpu* pSess_lofar_cpu = nullptr;
    CSession_lofar_gpu* pSess_lofar_gpu = nullptr;
    switch (INP_FORMAT)
    {
    case GUPPI:
        // pSess_guppi = new CSession_gpu_guppi(pPathInpFile, pPathOutFold, length_of_pulse, valD_max, sigma_Bound, lenWindow,lenChunk);
        // pSession = pSess_guppi;
        break;

    case FLOFAR:
        switch (PROCESSOR)
        {
        case CPU:
            pSess_lofar_cpu = new CSession_lofar_cpu(pPathInpFile, pPathOutFold, length_of_pulse
                , valD_min, valD_max, sigma_Bound, lenWindow, nbin, nfft);
            pSession = pSess_lofar_cpu;
            break;

        case GPU:
            pSess_lofar_gpu = new CSession_lofar_gpu(pPathInpFile, pPathOutFold, length_of_pulse
                , valD_min, valD_max, sigma_Bound, lenWindow, nbin, nfft);
            pSession = pSess_lofar_gpu;
            break;
        default: break;
        }
        break;

    default:
        return -1;

    }

    std::vector<std::vector<float>> vecImg;
    int  msamp = -1;
    if (-1 == pSession->launch(&vecImg, &msamp))
    {
        pSession = nullptr;

        if (pSess_lofar_cpu)
        {
            delete   pSess_lofar_cpu;
        }

        if (pSess_lofar_gpu)
        {
            delete   pSess_lofar_gpu;
        }
        return -1;
    }   
   

    int nImageRows = vecImg.size() * vecImg[0].size() / msamp;
    std::vector<float>vctData(vecImg.size() * vecImg[0].size());
    float* pntOut = vctData.data();
    for (int i = 0; i < vecImg.size(); ++i)
    {
        memcpy(&pntOut[i * vecImg[0].size()], vecImg[i].data(), vecImg[0].size() * sizeof(float));
    }
        std::array<long unsigned, 2> leshape101{ nImageRows, msamp };
    
        npy::SaveArrayAsNumpy("out_image.npy", false, leshape101.size(), leshape101.data(), vctData);
        #ifdef _WIN32 // Windows
        char filename_gpu[] = "image_gpu.png";
        createImg_(argc, argv, vctData, nImageRows, msamp, filename_gpu);
    #else
    #endif 
        

    if (pSession->m_pvctSuccessHeaders->size() > 0)
    {
        std::cout << "               Successful Chunk Numbers = " << pSession->m_pvctSuccessHeaders->size() << std::endl;
        //--------------------------------------

        char charrTemp[200] = { 0 };
        for (int i = 0; i < pSession->m_pvctSuccessHeaders->size(); ++i)
        {
            memset(charrTemp, 0, 200 * sizeof(char));
            (*(pSession->m_pvctSuccessHeaders))[i].createOutStr(charrTemp);
            std::cout << i + 1 << ". " << charrTemp << std::endl;
        }      
    }
    else
    {
        std::cout << "               Successful Chunk Were Not Detected= " << std::endl;
        return 0;
    }


    pSession = nullptr;
    if (pSess_lofar_cpu)
    {
        delete   pSess_lofar_cpu;
    }
    if (pSess_lofar_gpu)
    {
        delete   pSess_lofar_gpu;
    }


    char outputlogfile[300] = { 0 };
    strcpy(outputlogfile, "output.log");
    COutChunkHeader::writeReport(outputlogfile, pSession->m_pvctSuccessHeaders
        , length_of_pulse);

    


    //
    //    char chInp[200] = { 0 };
    //    std::cout << "if you  want to quit, print q" << endl;
    //    std::cout << "if you want to proceed, print y " << endl;
    //
    //    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    //    char ch0 = std::cin.get();
    //    if (ch0 == 'q')
    //    {
    //        return 0;
    //    }
    //    ////----------------------------------------------------------------------------------------------------
    //        // SECOND PART - PAINTINGS
    //    //----------------------------------------------------------------------------------------------------
    //    std::cout << "print number of chunk: " << endl;
    //    int numOrder = -1;
    //    std::cin >> numOrder;
    //    --numOrder;
    //
    //    int numBlock = -1;
    //    int numChunk = -1;
    //    long long lenChunk = -1;
    //    int n_fdmtRows = -1, n_fdmtCols = -1;
    //    int sucRow = -1, sucCol = -1, width = -1;
    //    float cohDisp = -1., snr = -1.;
    //
    //    COutChunkHeader::read_outputlogfile_line(outputlogfile
    //        , numOrder
    //        , &numBlock
    //        , &numChunk
    //        , &n_fdmtRows
    //        , &n_fdmtCols
    //        , &sucRow
    //        , &sucCol
    //        , &width
    //        , &cohDisp
    //        , &snr);
    //    //---------
    //    COutChunkHeader outChunkHeader(
    //        n_fdmtRows
    //        , n_fdmtCols
    //        , sucRow
    //        , sucCol
    //        , width
    //        , snr
    //        , cohDisp
    //        , numBlock - 1
    //        , numChunk - 1
    //    );
    //
    //   /* switch (INP_FORMAT)
    //    {
    //    case GUPPI:
    //        pSess_guppi = new CSession_gpu_guppi(pPathInpFile, pPathOutFold, length_of_pulse, valD_max, sigma_Bound, lenWindow);
    //        pSession = pSess_guppi;
    //        break;
    //
    //    case FLOFAR:
    //        pSess_lofar = new CSession_gpu_lofar(pPathInpFile, pPathOutFold, length_of_pulse, valD_max, sigma_Bound, lenWindow);
    //        pSession = pSess_lofar;
    //        break;
    //
    //    default:
    //        return -1;
    //
    //    }*/
    //
    //
    //    CFragment* pFRg = new CFragment();
    //    pSession->analyzeChunk(outChunkHeader, pFRg);
    //
    //  /*  pSession = NULL;
    //    if (pSess_guppi)
    //    {
    //        delete pSess_guppi;
    //    }
    //
    //    if (pSess_lofar)
    //    {
    //        pSess_lofar;
    //    }*/
    //
    //    int dim = pFRg->m_dim;
    //
    //
    //    std::array<long unsigned, 2> leshape101{ dim, dim };
    //
    //    npy::SaveArrayAsNumpy("out_image.npy", false, leshape101.size(), leshape101.data(), pFRg->m_vctData);
    //
    //    std::vector<float>v = pFRg->m_vctData;
    //    delete pFRg;
    //
    //#ifdef _WIN32 // Windows
    //    char filename_gpu[] = "image_gpu.png";
    //    createImg_(argc, argv, v, dim, dim, filename_gpu);
    //#else
    //#endif 
    

    return 0;
}
