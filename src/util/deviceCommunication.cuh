#ifndef DEVICE_COMMUNICATION_CUH
#define DEVICE_COMMUNICATION_CUH

#include "../mesh/deviceMesh.cuh"
#include "../mesh/mesh.cuh"

void allocateDeviceMemory(DeviceMesh* deviceMesh, int m, int n, int n0, int isQuad);
void copyHostToDeviceMesh(Mesh* hostMesh, DeviceMesh* deviceMesh);
Mesh copyDeviceMeshToHostMesh(DeviceMesh* deviceMesh);

#endif  // DEVICE_COMMUNICATION_CUH