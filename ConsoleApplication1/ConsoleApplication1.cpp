// ConsoleApplication1.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>

#include <vector>
#include <iostream>

using SizeType = int;
using DtGridType = std::vector<SizeType>;

int main() {
    // Example data
    std::vector<std::vector<DtGridType>> dt_grid = {
        { {1, 2}, {3, 4, 5} },
        { {6}, {7, 8}, {9, 10, 11} }
    };

    // Flattened vector of all integers
    std::vector<SizeType> flattened;
    // Vector to store the start indices of each DtGridType
    std::vector<SizeType> vctInnerRawPnts;
    // Vector to store the start indices of each vector<DtGridType>
    std::vector<SizeType> vctSubVectPnts;

    SizeType currentIndex = 0;

    for (const auto& subVect : dt_grid) {
        // Save the starting point of each subvector in dt_grid
        vctSubVectPnts.push_back(currentIndex);

        for (const auto& dtGrid : subVect) {
            // Save the starting point of each DtGridType
            vctInnerRawPnts.push_back(currentIndex);

            // Append elements of dtGrid to the flattened vector
            flattened.insert(flattened.end(), dtGrid.begin(), dtGrid.end());
            currentIndex += dtGrid.size();
        }
    }

    // Output the results
    std::cout << "Flattened vector: ";
    for (int num : flattened) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    std::cout << "vctInnerRawPnts indices: ";
    for (int index : vctInnerRawPnts) {
        std::cout << index << " ";
    }
    std::cout << std::endl;

    std::cout << "vctSubVectPnts indices: ";
    for (int index : vctSubVectPnts) {
        std::cout << index << " ";
    }
    std::cout << std::endl;

    return 0;
}
