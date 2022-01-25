#ifndef UTILS_CUH
#define UTILS_CUH

#include <stdio.h>
#include <stdlib.h>

#define MIN(A, B) (A > B ? B : A)

void printFloatArr(float* arr, int size);
void printIntArr(int* arr, int size);
int indexOfArr(int* arr, int size, int item);
void swap(int* a, int* b);

// Inspired by https://stackoverflow.com/a/14038590
#define cudaErrCheck(ans, message) \
    { cudaAssert((ans), (message), __FILE__, __LINE__); }

/**
 * @brief Checks for CUDA error codes. If the provided code is not cudaSuccess, it will output the error and exit the
 * program
 *
 * @param code The CUDA error code
 * @param message A custom error message provided by the developer
 * @param file The file in which the error occurred
 * @param line The line at which the error occurred
 */
inline void cudaAssert(cudaError_t code, const char* message, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "\n\nError: %s\n", message);
        // line -1, since checks are always done on the next line
        fprintf(stderr, "Cuda Error Message: %s %s %d\n\n", cudaGetErrorString(code), file, line);
        exit(-1);
    }
}

#endif  // UTILS_CUH