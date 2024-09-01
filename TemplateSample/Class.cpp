#include "Class.h"
template<typename T> T apb(T* a, T* b)
{
	return (*a) + (*b);
};
template int apb<int>(int* a, int* b);
