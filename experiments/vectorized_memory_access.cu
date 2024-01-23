#include <iostream>

int main() {
    const int SIZE = 4;
    float a[SIZE] = {1, 2, 3, 4};
    float b[SIZE];

    // Copying elements from array1 to array2
    reinterpret_cast<float4*>(b)[0] = reinterpret_cast<float4*>(a)[0]

    // Displaying elements of array2
    std::cout << "Elements of b: ";
    for (int i = 0; i < SIZE; ++i) {
        std::cout << b[i] << " ";
    }

    return 0;
}