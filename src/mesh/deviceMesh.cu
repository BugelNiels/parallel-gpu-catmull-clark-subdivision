#include <stdio.h>
#include <stdlib.h>

#include "../util/util.cuh"
#include "devicemesh.cuh"

DeviceMesh createEmptyCopyOnDevice(Mesh* mesh) {
    return initEmptyDeviceMesh(mesh->numVerts, mesh->numHalfEdges, mesh->numFaces, mesh->numEdges);
}

void setDevicePointerValue(int** loc, int val) {
    cudaError_t cuda_ret;
    cuda_ret = cudaMalloc((void**)loc, sizeof(int));
    cudaErrCheck(cuda_ret, "Unable to allocate device int pointer val");
    cuda_ret = cudaMemcpy(*loc, &val, sizeof(int), cudaMemcpyHostToDevice);
    cudaErrCheck(cuda_ret, "Unable to copy val to device pointer");
}

DeviceMesh initEmptyDeviceMesh(int numVerts, int numHalfEdges, int numFaces, int numEdges) {
    DeviceMesh mesh = {};
    mesh.numVerts = numVerts;
    mesh.numHalfEdges = numHalfEdges;
    mesh.numFaces = numFaces;
    mesh.numEdges = numEdges;
    return mesh;
}

DeviceMesh* toDevicePointer(DeviceMesh* mesh_h) {
    cudaError_t cuda_ret;
    DeviceMesh* mesh_d;
    cuda_ret = cudaMalloc((void**)&mesh_d, sizeof(DeviceMesh));
    cudaErrCheck(cuda_ret, "Unable to allocate device struct val");
    cuda_ret = cudaMemcpy(mesh_d, mesh_h, sizeof(DeviceMesh), cudaMemcpyHostToDevice);
    cudaErrCheck(cuda_ret, "Unable to copy struct to device pointer");
    return mesh_d;
}

DeviceMesh devicePointerToHostMesh(DeviceMesh* mesh_d) {
    cudaError_t cuda_ret;
    DeviceMesh mesh_h = {};
    cuda_ret = cudaMemcpy(&mesh_h, mesh_d, sizeof(DeviceMesh), cudaMemcpyDeviceToHost);
    cudaErrCheck(cuda_ret, "Unable to copy struct to host pointer");
    return mesh_h;
}

void freeDeviceMesh(DeviceMesh* mesh) {
    // Mesh is device pointer
    cudaFree(mesh->xCoords);
    cudaFree(mesh->yCoords);
    cudaFree(mesh->zCoords);
    cudaFree(mesh->twins);
    cudaFree(mesh->nexts);
    cudaFree(mesh->prevs);
    cudaFree(mesh->verts);
    cudaFree(mesh->edges);
    cudaFree(mesh->faces);
}
