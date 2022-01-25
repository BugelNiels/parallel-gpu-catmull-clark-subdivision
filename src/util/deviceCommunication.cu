#include <stdio.h>
#include <stdlib.h>

#include "deviceCommunication.cuh"
#include "util.cuh"

/**
 * @brief Allocates device memory for a non-quad mesh
 *
 * @param deviceMesh DeviceMesh in which to allocate memory
 * @param fD Number of faces at subdivision level D
 * @param hD Number of half-edges at subdivision level D
 * @param h0 Number of half-edges at subdivision level 0
 */
void allocateDeviceMemory(DeviceMesh* deviceMesh, int fD, int hD, int h0) {
    cudaError_t cuda_ret;
    // only allocate enough for the very first mesh
    cuda_ret = cudaMalloc((void**)&deviceMesh->nexts, h0 * sizeof(int));
    cudaErrCheck(cuda_ret, "Unable to allocate device memory for next array");
    cuda_ret = cudaMalloc((void**)&deviceMesh->prevs, h0 * sizeof(int));
    cudaErrCheck(cuda_ret, "Unable to allocate device memory for prev array");
    cuda_ret = cudaMalloc((void**)&deviceMesh->faces, h0 * sizeof(int));
    cudaErrCheck(cuda_ret, "Unable to allocate device memory for face array");

    allocateDeviceMemoryQuad(deviceMesh, fD, hD);
}

/**
 * @brief Allocates device memory for a quad mesh
 *
 * @param deviceMesh DeviceMesh in which to allocate memory
 * @param fD Number of faces at subdivision level D
 * @param hD Number of half-edges at subdivision level D
 */
void allocateDeviceMemoryQuad(DeviceMesh* deviceMesh, int fD, int hD) {
    cudaError_t cuda_ret;
    cuda_ret = cudaMalloc((void**)&deviceMesh->xCoords, fD * sizeof(float));
    cudaErrCheck(cuda_ret, "Unable to allocate device memory for X coordinates");
    cuda_ret = cudaMalloc((void**)&deviceMesh->yCoords, fD * sizeof(float));
    cudaErrCheck(cuda_ret, "Unable to allocate device memory for Y coordinates");
    cuda_ret = cudaMalloc((void**)&deviceMesh->zCoords, fD * sizeof(float));
    cudaErrCheck(cuda_ret, "Unable to allocate device memory for Z coordinates");

    cuda_ret = cudaMalloc((void**)&deviceMesh->twins, hD * sizeof(int));
    cudaErrCheck(cuda_ret, "Unable to allocate device memory for twin array");
    cuda_ret = cudaMalloc((void**)&deviceMesh->verts, hD * sizeof(int));
    cudaErrCheck(cuda_ret, "Unable to allocate device memory for vert array");
    cuda_ret = cudaMalloc((void**)&deviceMesh->edges, hD * sizeof(int));
    cudaErrCheck(cuda_ret, "Unable to allocate device memory for edge array");
}

/**
 * @brief Copies data from the host mesh to the device mesh.
 *
 * @param from Hot Mesh to copy the data from
 * @param to Device Mesh to copy the data to
 */
void copyHostToDeviceMesh(Mesh* from, DeviceMesh* to) {
    cudaError_t cuda_ret;
    int m = from->numVerts;
    cuda_ret = cudaMemcpy(to->xCoords, from->xCoords, m * sizeof(float), cudaMemcpyHostToDevice);
    cudaErrCheck(cuda_ret, "Unable to copy x-coordinates to the device");
    cuda_ret = cudaMemcpy(to->yCoords, from->yCoords, m * sizeof(float), cudaMemcpyHostToDevice);
    cudaErrCheck(cuda_ret, "Unable to copy y-coordinates to the device");
    cuda_ret = cudaMemcpy(to->zCoords, from->zCoords, m * sizeof(float), cudaMemcpyHostToDevice);
    cudaErrCheck(cuda_ret, "Unable to copy z-coordinates to the device");

    int n = from->numHalfEdges;
    cuda_ret = cudaMemcpy(to->twins, from->twins, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaErrCheck(cuda_ret, "Unable to copy twins to the device");
    cuda_ret = cudaMemcpy(to->verts, from->verts, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaErrCheck(cuda_ret, "Unable to copy verts to the device");
    cuda_ret = cudaMemcpy(to->edges, from->edges, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaErrCheck(cuda_ret, "Unable to copy edges to the device");

    if (from->isQuad == 0) {
        cuda_ret = cudaMemcpy(to->nexts, from->nexts, n * sizeof(int), cudaMemcpyHostToDevice);
        cudaErrCheck(cuda_ret, "Unable to copy nexts to the device");
        cuda_ret = cudaMemcpy(to->prevs, from->prevs, n * sizeof(int), cudaMemcpyHostToDevice);
        cudaErrCheck(cuda_ret, "Unable to copy prevs to the device");
        cuda_ret = cudaMemcpy(to->faces, from->faces, n * sizeof(int), cudaMemcpyHostToDevice);
        cudaErrCheck(cuda_ret, "Unable to copy faces to the device");
    }
}

/**
 * @brief Copies data from a device mesh to a new host mesh
 *
 * @param from Device Mesh to copy the data from
 * @return Mesh A new host mesh with the data from the device mesh
 */
Mesh copyDeviceMeshToHostMesh(DeviceMesh* from) {
    cudaError_t cuda_ret;

    printf("\tCopying mesh from device back to host...\n");

    Mesh to = initMesh(from->numVerts, from->numHalfEdges, from->numFaces, from->numEdges);
    to.isQuad = 1;
    allocQuadMesh(&to);

    // to already has the correct values for numVerts, numFaces etc.
    int vd = to.numVerts;
    cuda_ret = cudaMemcpy(to.xCoords, from->xCoords, vd * sizeof(float), cudaMemcpyDeviceToHost);
    cudaErrCheck(cuda_ret, "Unable to copy x-coordinates from the device");
    cuda_ret = cudaMemcpy(to.yCoords, from->yCoords, vd * sizeof(float), cudaMemcpyDeviceToHost);
    cudaErrCheck(cuda_ret, "Unable to copy y-coordinates from the device");
    cuda_ret = cudaMemcpy(to.zCoords, from->zCoords, vd * sizeof(float), cudaMemcpyDeviceToHost);
    cudaErrCheck(cuda_ret, "Unable to copy z-coordinates from the device");

    int hd = to.numHalfEdges;
    cuda_ret = cudaMemcpy(to.twins, from->twins, hd * sizeof(int), cudaMemcpyDeviceToHost);
    cudaErrCheck(cuda_ret, "Unable to copy twins from the device");
    cuda_ret = cudaMemcpy(to.verts, from->verts, hd * sizeof(int), cudaMemcpyDeviceToHost);
    cudaErrCheck(cuda_ret, "Unable to copy verts from the device");
    cuda_ret = cudaMemcpy(to.edges, from->edges, hd * sizeof(int), cudaMemcpyDeviceToHost);
    cudaErrCheck(cuda_ret, "Unable to copy edges from the device");
    printf("\tCopy completed: final mesh has %d faces\n", from->numFaces);
    return to;
}