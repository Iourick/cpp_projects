#pragma once
#include "Constants.h"
void  findMaxMinOfArray(float* d_arrfdmt_norm, unsigned const int LEn, float* valmax, float* valmin
	, unsigned int* iargmax, unsigned int* iargmin);

void  findMaxMinOfArray_(fdmt_type_* d_arrfdmt_norm, unsigned const int LEn, float* valmax, float* valmin
	, unsigned int* iargmax, unsigned int* iargmin);

unsigned long long ceil_power2_(const unsigned long long n);

int calc_len_sft(const float chanBW, const double pulse_length);

int calc_n_coherent(const double Fmin, const double Fmax, const unsigned int nchan, const double d_max
	, const double pulse_length);
