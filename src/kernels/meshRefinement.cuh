#ifndef MESH_REFINEMENT_CUH
#define MESH_REFINEMENT_CUH

#include "../mesh/deviceMesh.cuh"

#define BLOCK_SIZE 128
#define MAX_GRID_SIZE 262144
#define WARP_SIZE 32
#define FACES_PER_BLOCK (BLOCK_SIZE / 4)

__global__ void resetMesh(DeviceMesh* in, DeviceMesh* out);
__global__ void refineTopology(DeviceMesh* in, DeviceMesh* out);
__global__ void facePoints(DeviceMesh* in, DeviceMesh* out);
__global__ void edgePoints(DeviceMesh* in, DeviceMesh* out);
__global__ void vertexPoints(DeviceMesh* in, DeviceMesh* out);

#endif  // MESH_REFINEMENT_CUH