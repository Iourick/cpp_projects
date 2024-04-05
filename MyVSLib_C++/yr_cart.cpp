#include "yr_cart.h"
#include <cmath>
#include  <cfloat>


//------------------------------------------------
void  findMaxMinOfArray(float* d_arrfdmt_norm, unsigned const int LEn, float* valmax, float* valmin
	, unsigned int* iargmax, unsigned int* iargmin)
{
	*valmax = -FLT_MAX;
	*valmin = FLT_MAX;
	*iargmax = -1;
	*iargmin = -1;

	for (int i = 0; i < LEn; ++i)
	{
		if (d_arrfdmt_norm[i] > (*valmax))
		{
			*valmax = d_arrfdmt_norm[i];
			*iargmax = i;
		}
		if (d_arrfdmt_norm[i] < (*valmin))
		{
			*valmin = d_arrfdmt_norm[i];
			*iargmin = i;

		}
	}
}
//------------------------------------------------
void  findMaxMinOfArray_(fdmt_type_* d_arrfdmt_norm, unsigned const int LEn, float* valmax, float* valmin
	, unsigned int* iargmax, unsigned int* iargmin)
{
	*valmax = -FLT_MAX;
	*valmin = FLT_MAX;
	*iargmax = -1;
	*iargmin = -1;

	for (int i = 0; i < LEn; ++i)
	{
		if (d_arrfdmt_norm[i] > (*valmax))
		{
			*valmax = (float)d_arrfdmt_norm[i];
			*iargmax = i;
		}
		if (d_arrfdmt_norm[i] < (*valmin))
		{
			*valmin = (float)d_arrfdmt_norm[i];
			*iargmin = i;

		}
	}
}
//-------------------------------------------
unsigned long long ceil_power2_(const unsigned long long n)
{
	unsigned long long irez = 1;
	for (int i = 0; i < 63; ++i)
	{
		if (irez >= n)
		{
			return irez;
		}
		irez = irez << 1;
	}
	return -1;
}
//----------------------------
int calc_len_sft(const float chanBW, const double pulse_length)
{
	
	return ceil_power2_(ceil(pulse_length * 1.0E6 * chanBW));

}
//------------------------------------------
int calc_n_coherent(const double Fmin, const double Fmax, const unsigned int nchan, const double d_max 
	, const double pulse_length)
{
	double td = 4148.8 * (1. / (Fmin * Fmin) - 1. / (Fmax * Fmax)) * d_max ;
	double tau_telescope = 1.0E-6 / (Fmax - Fmin) *((float)nchan);
	//double val_N_d = td / tau_telescope;
	//double val_n_p = pulse_length / tau_telescope;
	return int(ceil(td * tau_telescope/ pulse_length/ pulse_length));
}
