#include "list.cuh"
#include "stdlib.h"
#include "util.cuh"

/**
 * @brief 
 * 
 * @return List 
 */
List initEmptyList() {
    List list;
    list.size = 10;
    list.arr = (int*)malloc(list.size * sizeof(int));
    list.i = 0;
    return list;
}

/**
 * @brief 
 * 
 * @param list 
 * @param item 
 */
void append(List* list, int item) {
    if (list->i == list->size) {
        list->size *= 2;
        list->arr = (int*)realloc(list->arr, list->size * sizeof(int));
    }
    list->arr[list->i] = item;
    list->i++;
}

int indexOf(List* list, int item) { return indexOfArr(list->arr, list->i, item); }
int listSize(List* list) { return list->i; }