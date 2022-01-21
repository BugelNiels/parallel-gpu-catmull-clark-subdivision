#include "objFile.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define INITIAL_LENGTH 128

// https://stackoverflow.com/a/58244503
// custom implementation, otherwise it won't work on windows :/
char *stringSep(char **stringp, const char *delim) {
    char *rv = *stringp;
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

void printObjFile(ObjFile obj) {
    for(int i = 0; i < obj.numVerts; i++) {
        printf("v %lf %lf %lf\n", obj.xCoords[i], obj.yCoords[i], obj.zCoords[i]);
    }

    printf("\n");
    for(int i = 0; i < obj.numFaces; i++) {
        printf("f");
        for(int j = 0; j < obj.faceValencies[i]; j++) {
            printf(" %d", obj.faceIndices[i][j]);
        }
        printf("\n");
    }
}

void addVertex(char* line, ObjFile* obj, int* vSize, int* v) {
    char* lineToParse = (char*)malloc((strlen(line) + 1) * sizeof(char));
    char* start = lineToParse;
    strcpy(lineToParse, line);
    // remove the v
    stringSep(&lineToParse, " ");
    obj->xCoords[*v] = atof(stringSep(&lineToParse, " "));
    obj->yCoords[*v] = atof(stringSep(&lineToParse, " "));
    obj->zCoords[*v] = atof(stringSep(&lineToParse, " "));
    (*v)++;

    if(*v >= *vSize - 4) {
        *vSize *= 2;
        obj->xCoords =  (float*)realloc(obj->xCoords, *vSize * sizeof(float));
        obj->yCoords =  (float*)realloc(obj->yCoords, *vSize * sizeof(float));
        obj->zCoords =  (float*)realloc(obj->zCoords, *vSize * sizeof(float));
    }
    free(start);
}

void addFace(char* line, ObjFile* obj, int* fSize, int* f) {
    char* lineToParse = (char*)malloc((strlen(line) + 1) * sizeof(char));
    char* start = lineToParse;
    strcpy(lineToParse, line);
    // remove the f
    stringSep(&lineToParse, " ");
    int currentSize = 4;
    int* indices = (int*)malloc(currentSize * sizeof(int));
    int i = 0;
    
    char *token;
    while((token = stringSep(&lineToParse, " "))) {   
        if(i >= currentSize) {
            currentSize *= 2;
            indices = (int*)realloc(indices, currentSize * sizeof(int));
        }
        indices[i] = atoi(token) - 1;
        i++;
    }
    obj->faceIndices[*f] = indices;
    if(i != 4) {
        obj->isQuad = 0;
    }
    obj->faceValencies[*f] = i;
    (*f)++;
    if(*f == *fSize) {
        *fSize *= 2;
        obj->faceIndices = (int**)realloc(obj->faceIndices, *fSize * sizeof(int*));
        obj->faceValencies = (int*)realloc(obj->faceValencies, *fSize * sizeof(int));
    }
    free(start);
}

void parseLine(char* line, ObjFile* obj, int* vSize, int* v, int* fSize, int* f) {
    if(strlen(line) <= 1) {
        return;
    }
    if(line[1] != ' ') {
        return;
    }
    char start = line[0];
    if(start == 'v') {
        addVertex(line, obj, vSize, v);
    } else if(start == 'f') {  
        addFace(line, obj, fSize, f);
    }
}

ObjFile readObjFromFile(char const* objFileName) {
    FILE *objFile = fopen(objFileName, "r");
    if (objFile == NULL)
    {
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

    int v = 0;
    int f = 0;
    while (fgets(line, len, objFile)) {
        parseLine(line, &obj, &vSize, &v, &fSize, &f);
    }
    obj.numVerts = v;
    obj.numFaces = f;
    printf("%d %d \n", v, f);
    if(obj.isQuad == 1) {
        printf("Loaded quad mesh.\n");
    } else {
        printf("Loaded non-quad mesh.\n");
    }
    fclose(objFile);
    free(line);
    return obj;
}

void freeObjFile(ObjFile objFile) {

    free(objFile.xCoords);
    free(objFile.yCoords);
    free(objFile.zCoords);

    for(int f = 0; f < objFile.numFaces; f++) {
        free(objFile.faceIndices[f]);
    }
    free(objFile.faceIndices);
    free(objFile.faceValencies);
}