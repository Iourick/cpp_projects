// ConsoleApplication1.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>

#include <span>
#include <vector>
#include <iostream>

void print_span(std::span<int> s) {
    for (int value : s) {
        std::cout << value << " ";
    }
    std::cout << "\n";
}

int main() {
    // Example 1: Using a vector
    std::vector<int> vec = { 1, 2, 3, 4, 5 };
    std::span<int> vec_span(vec);
    print_span(vec_span);

    // Example 2: Using a static array
    int arr[] = { 10, 20, 30, 40, 50 };
    std::span<int> arr_span(arr);
    print_span(arr_span);

    // Example 3: Using a subrange of an array
    std::span<int> subrange(arr, 3);
    print_span(subrange);

    return 0;
}

