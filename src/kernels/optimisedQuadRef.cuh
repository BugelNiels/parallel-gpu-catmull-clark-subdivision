#ifndef OPTIMISED_QUAD_REFINEMENT_CUH
#define OPTIMISED_QUAD_REFINEMENT_CUH

#include "meshRefinement.cuh"
#include "quadRefinement.cuh"

__global__ void setBoundaryVerts(DeviceMesh* in);
__global__ void optimisedSubdivide(DeviceMesh* in, DeviceMesh* out, int v0);

#endif  // OPTIMISED_QUAD_REFINEMENT_CUH