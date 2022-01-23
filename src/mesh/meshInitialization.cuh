#ifndef MESH_INITIALIZATION_CUH
#define MESH_INITIALIZATION_CUH

#include <stdio.h>
#include <stdlib.h>

#include "mesh.cuh"
#include "objFile.cuh"

Mesh meshFromObjFile(ObjFile obj);

#endif  // MESH_INITIALIZATION_CUH