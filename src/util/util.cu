#include <stdio.h>
#include <stdlib.h>

#include "util.cuh"

void printFloatArr(float* arr, int size) {
    if (size == 0) {
        return;
    }
    for (int i = 0; i < size - 1; i++) {
        printf("%lf, ", arr[i]);
    }
    printf("%lf\n\n", arr[size - 1]);
}

void printIntArr(int* arr, int size) {
    if (size == 0) {
        return;
    }
    for (int i = 0; i < size - 1; i++) {
        printf("%d ", arr[i]);
    }
    printf("%d\n\n", arr[size - 1]);
}

int indexOfArr(int* arr, int size, int item) {
    for (int i = 0; i < size; i++) {
        if (arr[i] == item) {
            return i;
        }
    }
    return -1;
}
