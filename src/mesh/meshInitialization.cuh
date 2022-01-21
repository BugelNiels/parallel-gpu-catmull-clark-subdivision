#ifndef MESH_INITIALIZATION_CUH
#define MESH_INITIALIZATION_CUH

#include <stdlib.h>
#include <stdio.h>

#include "mesh.cuh"
#include "objFile.cuh"

Mesh meshFromObjFile(ObjFile obj);

#endif // MESH_INITIALIZATION_CUH