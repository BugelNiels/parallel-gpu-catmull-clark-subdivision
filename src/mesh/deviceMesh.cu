#include <stdio.h>
#include <stdlib.h>

#include "../util/util.cuh"
#include "devicemesh.cuh"

/**
 * @brief Creates an empty copy of the provided mesh on the device.
 * Only copies the number of vertices, half-edges, faces and edges.
 *
 * @param mesh The mesh whose properties to copy
 * @return DeviceMesh The empty device mesh
 */
DeviceMesh createEmptyCopyOnDevice(Mesh* mesh) {
    return initEmptyDeviceMesh(mesh->numVerts, mesh->numHalfEdges, mesh->numFaces, mesh->numEdges);
}

/**
 * @brief Initializes an empty device mesh with the given properties
 *
 * @param numVerts Number of vertices
 * @param numHalfEdges Number of half-edges
 * @param numFaces Number of faces
 * @param numEdges Number of edges
 * @return DeviceMesh An empty device mesh
 */
DeviceMesh initEmptyDeviceMesh(int numVerts, int numHalfEdges, int numFaces, int numEdges) {
    DeviceMesh mesh = {};
    mesh.numVerts = numVerts;
    mesh.numHalfEdges = numHalfEdges;
    mesh.numFaces = numFaces;
    mesh.numEdges = numEdges;
    return mesh;
}

/**
 * @brief Copies the host pointer that points to the device mesh to a device pointer that points to the device mesh
 *
 * @param mesh_h The host mesh to copy
 * @return DeviceMesh* The device pointer to the device mesh top copy
 */
DeviceMesh* toDevicePointer(DeviceMesh* mesh_h) {
    cudaError_t cuda_ret;
    DeviceMesh* mesh_d;
    cuda_ret = cudaMalloc((void**)&mesh_d, sizeof(DeviceMesh));
    cudaErrCheck(cuda_ret, "Unable to allocate device struct val");
    cuda_ret = cudaMemcpy(mesh_d, mesh_h, sizeof(DeviceMesh), cudaMemcpyHostToDevice);
    cudaErrCheck(cuda_ret, "Unable to copy struct to device pointer");
    return mesh_d;
}

/**
 * @brief Copies the device pointer that points to the device mesh to a host pointer that points to the device mesh
 *
 * @param mesh_d The device mesh to copy
 * @return DeviceMesh The device mesh
 */
DeviceMesh devicePointerToHostMesh(DeviceMesh* mesh_d) {
    cudaError_t cuda_ret;
    DeviceMesh mesh_h = {};
    cuda_ret = cudaMemcpy(&mesh_h, mesh_d, sizeof(DeviceMesh), cudaMemcpyDeviceToHost);
    cudaErrCheck(cuda_ret, "Unable to copy struct to host pointer");
    return mesh_h;
}

/**
 * @brief Frees memory of a device mesh
 *
 * @param mesh Device mesh to free
 */
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
