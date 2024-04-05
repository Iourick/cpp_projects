#include "utilites.h"
#include <fstream>
#include <iostream>

int readHeader(char* chInpFilePass, unsigned int &lenarr, unsigned int& n_p
    ,float& valD_max_, float& valf_min_, float& valf_max_, float& valSigmaBound_)
{
    FILE* file = fopen(chInpFilePass, "rb");
        if (file == nullptr) {
        std::cerr << "Error opening file." << std::endl;
        return 1;
    }


    // Read the integer variables
    fread(&lenarr, sizeof(int), 1, file);
    fread(&n_p, sizeof(int), 1, file);

    // Read the float variables
    fread(&valD_max_, sizeof(float), 1, file);
    fread(&valf_min_, sizeof(float), 1, file);
    fread(&valf_max_, sizeof(float), 1, file);
    fread(&valSigmaBound_, sizeof(float), 1, file);    
    fclose(file);
}
//------------------------------------------------------------------
