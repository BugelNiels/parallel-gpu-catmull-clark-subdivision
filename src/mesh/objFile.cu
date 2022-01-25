#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "objFile.cuh"

#define INITIAL_LENGTH 128

// https://stackoverflow.com/a/58244503
// custom implementation, otherwise it won't work on windows :/
/**
 * @brief Seperates a string by the specified delimiter. Taken from https://stackoverflow.com/a/58244503
 * Note that it modifies the provided string.
 * This is necessary for the program to work on windows.
 *
 * @param stringp String to seperate
 * @param delim Delimiter
 * @return char* Pointer to the next position of the delimiter in the string
 */
char* stringSep(char** stringp, const char* delim) {
    char* rv = *stringp;
    if (rv) {
        *stringp += strcspn(*stringp, delim);
        if (**stringp) {
            *(*stringp)++ = '\0';
        } else {
            *stringp = 0;
        }
    }
    return rv;
}

/**
 * @brief Adds a vertex to the objFile based on the provided line
 *
 * @param line The line in which the vertex data is located
 * @param obj The ObjFile to add the new vertex to
 * @param vSize The current size of the vertex coordinate array in the ObjFile
 */
void addVertex(char* line, ObjFile* obj, int* vSize) {
    char* lineToParse = (char*)malloc((strlen(line) + 1) * sizeof(char));
    char* start = lineToParse;
    strcpy(lineToParse, line);
    // remove the v
    stringSep(&lineToParse, " ");
    obj->xCoords[obj->numVerts] = atof(stringSep(&lineToParse, " "));
    obj->yCoords[obj->numVerts] = atof(stringSep(&lineToParse, " "));
    obj->zCoords[obj->numVerts] = atof(stringSep(&lineToParse, " "));
    obj->numVerts++;

    if (obj->numVerts >= *vSize - 4) {
        *vSize *= 2;
        obj->xCoords = (float*)realloc(obj->xCoords, *vSize * sizeof(float));
        obj->yCoords = (float*)realloc(obj->yCoords, *vSize * sizeof(float));
        obj->zCoords = (float*)realloc(obj->zCoords, *vSize * sizeof(float));
    }
    free(start);
}

/**
 * @brief Adds a face to the objFile based on the provided line
 *
 * @param line The line in which the face data is located
 * @param obj The ObjFile to add the new face to
 * @param fSize The current size of faceIndices and faceValencies arrays
 */
void addFace(char* line, ObjFile* obj, int* fSize) {
    char* lineToParse = (char*)malloc((strlen(line) + 1) * sizeof(char));
    char* start = lineToParse;
    strcpy(lineToParse, line);
    // remove the f
    stringSep(&lineToParse, " ");
    int currentSize = 4;
    int* indices = (int*)malloc(currentSize * sizeof(int));
    int i = 0;

    char* token;
    // Add every vertex index to the indices array
    while ((token = stringSep(&lineToParse, " "))) {
        if (i >= currentSize) {
            currentSize *= 2;
            indices = (int*)realloc(indices, currentSize * sizeof(int));
        }
        indices[i] = atoi(token) - 1;
        i++;
    }
    obj->faceIndices[obj->numFaces] = indices;
    if (i != 4) {
        obj->isQuad = 0;
    }
    obj->faceValencies[obj->numFaces] = i;
    obj->numFaces++;
    if (obj->numFaces == *fSize) {
        *fSize *= 2;
        obj->faceIndices = (int**)realloc(obj->faceIndices, *fSize * sizeof(int*));
        obj->faceValencies = (int*)realloc(obj->faceValencies, *fSize * sizeof(int));
    }
    free(start);
}

/**
 * @brief Parses a line from aa .obj file
 *
 * @param line The line to parse
 * @param obj The ObjFile to which to add the data
 * @param vSize The current size of the vertex coordinate array in the ObjFile
 * @param fSize The current size of faceIndices and faceValencies arrays
 */
void parseLine(char* line, ObjFile* obj, int* vSize, int* fSize) {
    if (strlen(line) <= 1) {
        return;
    }
    if (line[1] != ' ') {
        return;
    }
    char start = line[0];
    if (start == 'v') {
        addVertex(line, obj, vSize);
    } else if (start == 'f') {
        addFace(line, obj, fSize);
    }
}

/**
 * @brief Parses a .obj file and puts the result in the ObjFile struct
 *
 * @param path The path to the .obj file
 * @return ObjFile Struct containing data from the .obj file
 */
ObjFile parseObjFile(char const* path) {
    FILE* objFile = fopen(path, "r");
    if (objFile == NULL) {
        printf("Error opening .obj file!\n");
        exit(1);
    }
    size_t len = INITIAL_LENGTH;
    char* line = (char*)malloc(len * sizeof(char));

    ObjFile obj;
    obj.isQuad = 1;

    int fSize = INITIAL_LENGTH;
    int vSize = INITIAL_LENGTH;
    obj.xCoords = (float*)malloc(vSize * sizeof(float));
    obj.yCoords = (float*)malloc(vSize * sizeof(float));
    obj.zCoords = (float*)malloc(vSize * sizeof(float));

    obj.faceIndices = (int**)malloc(fSize * sizeof(int*));
    obj.faceValencies = (int*)malloc(fSize * sizeof(int));

    obj.numVerts = 0;
    obj.numFaces = 0;
    while (fgets(line, len, objFile)) {
        parseLine(line, &obj, &vSize, &fSize);
    }
    if (obj.isQuad == 1) {
        printf("Loaded quad mesh.\n");
    } else {
        printf("Loaded non-quad mesh.\n");
    }
    fclose(objFile);
    free(line);
    return obj;
}

/**
 * @brief Frees data from an ObjFile
 *
 * @param objFile The ObjFile whose data to free
 */
void freeObjFile(ObjFile objFile) {
    free(objFile.xCoords);
    free(objFile.yCoords);
    free(objFile.zCoords);

    for (int f = 0; f < objFile.numFaces; f++) {
        free(objFile.faceIndices[f]);
    }
    free(objFile.faceIndices);
    free(objFile.faceValencies);
}