#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cudaSubdivision.cuh"
#include "mesh/meshInitialization.cuh"
#include "mesh/objFile.cuh"

#define BUFFER_SIZE 80

/**
 * @brief Creates a .obj file path with the provided directory and name
 * 
 * @param dir The directory to prepend to the path. Should include "/" at the end
 * @param name The name of the .obj file
 * @return char* A .obj file path
 */
char* createObjFilePath(char const* dir, char const* name) {
    char* filePath = (char*)malloc(BUFFER_SIZE * sizeof(char));
    strcpy(filePath, dir);
    strcat(filePath, name);
    strcat(filePath, ".obj");
    return filePath;
}

/**
 * @brief Entry point of the CUDA subdivision program. Usage: 
 * 
 * @param argc Number of arguments
 * @param argv Program arguments
 * @return int Exit code
 */
int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf(
            " Please provide a subdivision level, source .obj file and optionally a destination .obj file.\n Do not "
            "include \".obj\" in the filenames.\n\n Example:\n \t./subdivide 2 bigguy bigguy_result\n\n");
        return 0;
    }
    int subdivisionLevel = atoi(argv[1]);
    if (subdivisionLevel < 1) {
        printf(" You must apply at least 1 level of subdivision.\n");
        return 0;
    }
    char* filePath = createObjFilePath("models/", argv[2]);
    char* resultPath = NULL;
    if (argc == 4) {
        resultPath = createObjFilePath("results/", argv[3]);
    }

    ObjFile objFile = parseObjFile(filePath);
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