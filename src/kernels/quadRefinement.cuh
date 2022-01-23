#ifndef QUAD_REFINEMENT_CUH
#define QUAD_REFINEMENT_CUH

#include "meshRefinement.cuh"

__device__ int valenceQuad(int h, DeviceMesh* in);
__global__ void quadRefineTopology(DeviceMesh* in, DeviceMesh* out);
__global__ void quadFacePoints(DeviceMesh* in, DeviceMesh* out);
__global__ void quadEdgePoints(DeviceMesh* in, DeviceMesh* out);
__global__ void quadVertexPoints(DeviceMesh* in, DeviceMesh* out);

#endif // QUAD_REFINEMENT_CUH