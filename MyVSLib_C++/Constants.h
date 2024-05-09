#pragma once
#include <cstdint>


// FLOP's quantity for typical tests in giga flops
const float GFLPS_512 = 0.00320386;
const float GFLPS_1024 = 0.0137001;
const float GFLPS_2048 = 0.0586616;
const float GFLPS_512_1024 = 0.006518622;

//---------------------------------------------
//#define  DISPERSION_CONSTANT  4.148808*10**9 // Mhz* pc ^ -1 * cm ^ 3
const long double  DISPERSION_CONSTANT = 4148808000.0;// Mhz* pc ^ -1 * cm ^ 3


#define fdmt_type_     float//int// 
#define inp_type_        char//float//