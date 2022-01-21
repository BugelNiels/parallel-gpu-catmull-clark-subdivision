#ifndef CUDA_SUBDIVISION_CUH
#define CUDA_SUBDIVISION_CUH

#include "mesh/mesh.cuh"

Mesh cudaSubdivide(Mesh* mesh, int subdivisionLevel);

#endif // CUDA_SUBDIVISION_CUH