#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cudaSubdivision.cuh"
#include "mesh/meshInitialization.cuh"
#include "mesh/objFile.cuh"

#define BUFFER_SIZE 80

/**
 * @brief Create a Obj File Path object
 * 
 * @param dir 
 * @param name 
 * @return char* 
 */
char* createObjFilePath(char const* dir, char const* name) {
    char* filePath = (char*)malloc(BUFFER_SIZE * sizeof(char));
    strcpy(filePath, dir);
    strcat(filePath, name);
    strcat(filePath, ".obj");
    return filePath;
}

/**
 * @brief 
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf(
            "Please provide a subdivision level, source .obj file and optionally a destination .obj file. Do not "
            "include \".obj\" in the filename\n");
        return 0;
    }
    int subdivisionLevel = atoi(argv[1]);
    if (subdivisionLevel < 1) {
        printf("Subdivision level must be at least 1");
        return 0;
    }
    char* filePath = createObjFilePath("models/", argv[2]);
    char* resultPath = NULL;
    if (argc == 4) {
        resultPath = createObjFilePath("results/", argv[3]);
    }

    ObjFile objFile = readObjFromFile(filePath);
    free(filePath);
    Mesh mesh = meshFromObjFile(objFile);
    freeObjFile(objFile);

    Mesh result = cudaSubdivide(&mesh, subdivisionLevel);
    freeMesh(&mesh);
    if (resultPath != NULL) {
        toObjFile(&result, resultPath);
    }
    freeMesh(&result);
    return 0;
}