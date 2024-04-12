#include "main.h"
#include <iostream>


#include <fstream>

#include "OutChunk.h"
#include <array>
#include <string>
#include <vector>
#include <cstdlib> // For random value generation
#include <ctime>   // For seeding the random number generator

#include <algorithm> 

#include <chrono>

#include "Constants.h"
#include "SessionB.h"
#include "Fragment.h"
#include "Session_lofar_cpu.h"

#define _CRT_SECURE_NO_WARNINGS
using namespace std;
using namespace std;



/************** DATA FOR LOFAR ****************************/
char PathInpFile[] = "D://BASSA//hdf5_data//L2012176_SAP000_B000_S0_P001_bf.h5";
char PathOutFold[] = "OutPutFold";
TYPE_OF_INP_FORMAT INP_FORMAT = FLOFAR;
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
   
    CSession_lofar_cpu* pSess_lofar = new CSession_lofar_cpu(pPathInpFile, pPathOutFold, length_of_pulse
        , valD_min, valD_max, sigma_Bound, lenWindow, nbin, nfft);
   
    if (-1 == pSess_lofar->launch())
    {       

        if (pSess_lofar)
        {
            delete   pSess_lofar;
        }
        return -1;
    }
    delete   pSess_lofar;
    
    return 0;
}
