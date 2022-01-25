#ifndef OBJ_FILE_CUH
#define OBJ_FILE_CUH

#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Struct that contains data from an ObjFile
 * 
 */
typedef struct ObjFile {
    // these have length numVerts
    float* xCoords;
    float* yCoords;
    float* zCoords;

    // Both have length numFaces
    int* faceValencies;
    // faceIndices[i] has length faceValencies[i]
    int** faceIndices;

    int numFaces;
    int numVerts;

    // 1 if the mesh is a quad; 0 otherwise
    int isQuad;
} ObjFile;

ObjFile parseObjFile(char const* path);
void freeObjFile(ObjFile objFile);

#endif  // OBJ_FILE_CUH