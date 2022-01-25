#include "list.cuh"
#include "stdlib.h"
#include "util.cuh"

/**
 * @brief Creates an empty list
 * 
 * @return List An empty list
 */
List initEmptyList() {
    List list;
    list.size = 10;
    list.arr = (int*)malloc(list.size * sizeof(int));
    list.i = 0;
    return list;
}

/**
 * @brief Appends an item to the end of the list
 * 
 * @param list The list to add the item to
 * @param item The item to add
 */
void append(List* list, int item) {
    if (list->i == list->size) {
        list->size *= 2;
        list->arr = (int*)realloc(list->arr, list->size * sizeof(int));
    }
    list->arr[list->i] = item;
    list->i++;
}

/**
 * @brief Returns the index of an item in the provided list. Returns -1 if not found
 * 
 * @param list The list in which to search for the item
 * @param item The item to find the index of
 * @return int The index of the item in the list. -1 if it does not exist
 */
int indexOf(List* list, int item) { return indexOfArr(list->arr, list->i, item); }

/**
 * @brief Returns the size of the list
 * 
 * @param list The list you want to know the size of
 * @return int The size of the provided list
 */
int listSize(List* list) { return list->i; }