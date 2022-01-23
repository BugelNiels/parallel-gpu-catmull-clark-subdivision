#ifndef KERNEL_UTILS_CUH
#define KERNEL_UTILS_CUH
#include "../../mesh/deviceMesh.cuh"

__device__ int cycleLength(int h, DeviceMesh* in);

__device__ int valence(int h, DeviceMesh* in);
__device__ int valenceQuad(int h, DeviceMesh* in);

inline __device__ int next(int h) { return h % 4 == 3 ? h - 3 : h + 1; }
inline __device__ int prev(int h) { return h % 4 == 0 ? h + 3 : h - 1; }
inline __device__ int face(int h) { return h / 4; }

#endif  // KERNEL_UTILS_CUH