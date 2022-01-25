#include <stdio.h>
#include <stdlib.h>

#include "util.cuh"

/**
 * @brief Prints an array of floats
 * 
 * @param arr The array to print
 * @param size The size of the array
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
 * @brief Prints an array of integers
 * 
 * @param arr The array to print
 * @param size The size of the array
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
 * @brief Finds the index of an item in an integer array. 
 * 
 * @param arr The array to search in
 * @param size The size of the array
 * @param item The item to search for
 * @return int The index of the item in the array; -1 if it does not exist
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
 * @brief Swaps to integers
 * 
 * @param a Integer a
 * @param b Integer b
 */
void swap(int* a, int* b) {
	int c = *a;
	*a = *b;
	*b = c;
}