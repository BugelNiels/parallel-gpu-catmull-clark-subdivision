#ifndef OBJ_FILE_CUH
#define OBJ_FILE_CUH

#include <stdlib.h>
#include <stdio.h>

typedef struct ObjFile {
	// these have length numVerts
	float* xCoords;
	float* yCoords;
	float* zCoords;

	// these all have length numHalfEdges
	int** faceIndices;
    int* faceValencies;

  	int numFaces;
  	int numVerts;

	int isQuad;
} ObjFile;

ObjFile readObjFromFile(char const* objFileName);
void freeObjFile(ObjFile objFile);

#endif // OBJ_FILE_CUH