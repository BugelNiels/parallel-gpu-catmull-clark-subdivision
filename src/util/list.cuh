#ifndef LIST_CUH
#define LIST_CUH

/**
 * @brief List data structure
 * 
 */
typedef struct List {
    int* arr;
    int size;
    int i;
} List;

List initEmptyList();
void append(List* list, int item);
int indexOf(List* list, int item);
int listSize(List* list);

#endif