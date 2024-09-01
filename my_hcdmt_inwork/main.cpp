#include "main.h"
#include <iostream>


#include <GL/glut.h>
#include <fstream>
#include "OutChunkHeader.h"

#include <array>
#include <string>
#include <vector>
#include <cstdlib> // For random value generation
#include <ctime>   // For seeding the random number generator
#include "npy.hpp"
#include <algorithm> 

#include <chrono>

#include "DrawImg.h"
#include "Constants.h"
#include "SessionB.h"
#include "Fragment.h"
#include "Session_lofar_cpu.h"
#include "Session_lofar_gpu.cuh"
#include "Session_guppi_cpu.h"
#include "Session_guppi_gpu.cuh"

#define _CRT_SECURE_NO_WARNINGS
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
float sigma_Bound = 4.0;
int lenWindow = 5;
int nbin = 262144 ;
int nfft =4;
/*************** ! DATA FOR LOFAR *****************************/


/***************   GUPPI ********************************************/
 //char PathInpFile[] = "D://weizmann//RAW_DATA//blc20_guppi_57991_49905_DIAG_FRB121102_0011.0007.raw";
 //char PathOutFold[] = "OutPutFold";
 //
 //TYPE_OF_INP_FORMAT INP_FORMAT = GUPPI;
 // TYPE_OF_PROCESSOR PROCESSOR = GPU;
 // char* pPathInpFile = PathInpFile;
 // char* pPathOutFold = PathOutFold;
 // double valD_min = 30.0;
 // double valD_max = 600.0;
 // double length_of_pulse = 3.41333333E-7 * 64.0;//;
 // float sigma_Bound = 12.;
 // int lenWindow = 1;
 // int nbin = 262144*2 ;
 // int nfft =1;
/*************** !  GUPPI ********************************************/


   

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
    CSession_guppi_cpu* pSess_guppi_cpu = nullptr;
    CSession_guppi_gpu* pSess_guppi_gpu = nullptr;
    switch (INP_FORMAT)
    {
        case GUPPI:
            switch (PROCESSOR)
            {
            case CPU:
                pSess_guppi_cpu = new CSession_guppi_cpu(pPathInpFile, pPathOutFold, length_of_pulse
                    , valD_min, valD_max, sigma_Bound, lenWindow, nbin, nfft);
                pSession = pSess_guppi_cpu;
                break;

            case GPU:
                pSess_guppi_gpu = new CSession_guppi_gpu(pPathInpFile, pPathOutFold, length_of_pulse
                    , valD_min, valD_max, sigma_Bound, lenWindow, nbin, nfft);
                pSession = pSess_guppi_gpu;
                break;
            default: break;
            }
                 
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
        if (pSess_guppi_cpu)
        {
            delete   pSess_guppi_cpu;
        }

        if (pSess_guppi_gpu)
        {
            delete   pSess_guppi_gpu;
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
    CSession_guppi_cpu* pSess_guppi_cpu = nullptr;
    CSession_guppi_gpu* pSess_guppi_gpu = nullptr;
    switch (INP_FORMAT)
    {
    case GUPPI:
        switch (PROCESSOR)
        {
        case CPU:
            pSess_guppi_cpu = new CSession_guppi_cpu(pPathInpFile, pPathOutFold, length_of_pulse
                , valD_min, valD_max, sigma_Bound, lenWindow, nbin, nfft);
            pSession = pSess_guppi_cpu;
            break;

        case GPU:
            pSess_guppi_gpu = new CSession_guppi_gpu(pPathInpFile, pPathOutFold, length_of_pulse
                , valD_min, valD_max, sigma_Bound, lenWindow, nbin, nfft);
            pSession = pSess_guppi_gpu;
            break;
        default: break;
        }

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
        if (pSess_guppi_cpu)
        {
            delete   pSess_guppi_cpu;
        }

        if (pSess_guppi_gpu)
        {
            delete   pSess_guppi_gpu;
        }
        return -1;
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
    if (pSess_guppi_cpu)
    {
        delete   pSess_guppi_cpu;
    }

    if (pSess_guppi_gpu)
    {
        delete   pSess_guppi_gpu;
    }
    if (vecImg.size() == 0)
    {
        return 0;
    }
    else
    {
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


    pSession = nullptr;
    if (pSess_lofar_cpu)
    {
        delete   pSess_lofar_cpu;
    }
    if (pSess_lofar_gpu)
    {
        delete   pSess_lofar_gpu;
    }  
  
    return 0;
}
