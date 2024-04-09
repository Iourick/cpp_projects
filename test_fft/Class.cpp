#include "Class.h"
#include <cstdlib>
#include <cstring>
//--------------------------------------------------------------

template <typename T>
void Class::roll_(T* arr, const int lenarr, const int ishift)
{
    if (ishift == 0)
    {
        return;
    }
    T* arr0 = (T*)malloc(ishift * sizeof(T));
    T* arr1 = (T*)malloc((lenarr - ishift) * sizeof(T));
    memcpy(arr0, arr + lenarr - 1 - ishift, ishift * sizeof(T));
    memcpy(arr1, arr, (lenarr - ishift) * sizeof(T));
    memcpy(arr, arr0, ishift * sizeof(T));
    memcpy(arr + ishift, arr1, (lenarr - ishift) * sizeof(T));
    free(arr0);
    free(arr1);
}
template void Class::roll_<int>(int* arr, const int lenarr, const int ishift);
template void Class::roll_<float>(float* arr, const int lenarr, const int ishift);