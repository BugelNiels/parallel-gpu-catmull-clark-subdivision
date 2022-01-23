#ifndef KERNEL_INVOKER_CUH
#define KERNEL_INVOKER_CUH

#include "mesh/deviceMesh.cuh"

DeviceMesh performSubdivision(DeviceMesh* in, DeviceMesh* out, int subdivisionLevel, Mesh* mesh);

#endif  // KERNEL_INVOKER_CUH