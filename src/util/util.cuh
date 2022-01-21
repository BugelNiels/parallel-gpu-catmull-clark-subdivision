#ifndef UTILS_CUH
#define UTILS_CUH

#include <stdlib.h>
#include <stdio.h>

#define MIN(A, B) (A > B ? B : A)

void printFloatArr(float* arr, int size);
void printIntArr(int* arr, int size);
int indexOfArr(int* arr, int size, int item);

#define FATAL(msg, ...)                                                      \
  do {                                                                       \
    fprintf(stderr, "[%s:%d] " msg "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
    exit(-1);                                                                \
  } while (0)

  
// Inspired by https://stackoverflow.com/a/14038590  
#define cudaErrCheck(ans, message) { cudaAssert((ans), (message), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char * message, const char *file, int line) {
   if (code != cudaSuccess) {
      fprintf(stderr, "\n\nError: %s\n", message); 
      // line -1, since checks are always done on the next line
      fprintf(stderr, "Cuda Error Message: %s %s %d\n\n", cudaGetErrorString(code), file, line - 1);
      exit(-1);
   }
}


#endif // UTILS_CUH