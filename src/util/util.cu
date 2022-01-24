#include <stdio.h>
#include <stdlib.h>

#include "util.cuh"

/**
 * @brief 
 * 
 * @param arr 
 * @param size 
 */
void printFloatArr(float* arr, int size) {
    if (size == 0) {
        return;
    }
    for (int i = 0; i < size - 1; i++) {
        printf("%lf, ", arr[i]);
    }
    printf("%lf\n\n", arr[size - 1]);
}

/**
 * @brief 
 * 
 * @param arr 
 * @param size 
 */
void printIntArr(int* arr, int size) {
    if (size == 0) {
        return;
    }
    for (int i = 0; i < size - 1; i++) {
        printf("%d ", arr[i]);
    }
    printf("%d\n\n", arr[size - 1]);
}

/**
 * @brief 
 * 
 * @param arr 
 * @param size 
 * @param item 
 * @return int 
 */
int indexOfArr(int* arr, int size, int item) {
    for (int i = 0; i < size; i++) {
        if (arr[i] == item) {
            return i;
        }
    }
    return -1;
}

/**
 * @brief 
 * 
 * @param a 
 * @param b 
 */
void swap(int* a, int* b) {
	int c = *a;
	*a = *b;
	*b = c;
}