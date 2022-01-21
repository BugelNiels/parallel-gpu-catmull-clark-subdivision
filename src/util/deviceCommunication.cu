#include <stdlib.h>
#include <stdio.h>

#include "deviceCommunication.cuh"
#include "util.cuh"

// m = number of vertices in vD; n = number of half edges in vD
void allocateDeviceMemory(DeviceMesh* deviceMesh, int m, int n, int n0, int isQuad) {
	cudaError_t cuda_ret;
	printf("Allocating device variables\n");
    cuda_ret = cudaMalloc((void**)&deviceMesh->xCoords, m * sizeof(float));
    cudaErrCheck(cuda_ret, "Unable to allocate device memory for X coordinates");
    cuda_ret = cudaMalloc((void**)&deviceMesh->yCoords, m * sizeof(float));
    cudaErrCheck(cuda_ret, "Unable to allocate device memory for Y coordinates");
	cuda_ret = cudaMalloc((void**)&deviceMesh->zCoords, m * sizeof(float));
    cudaErrCheck(cuda_ret, "Unable to allocate device memory for Z coordinates");  


	cuda_ret = cudaMalloc((void**)&deviceMesh->twins, n * sizeof(int));
    cudaErrCheck(cuda_ret, "Unable to allocate device memory for twin array");
	cuda_ret = cudaMalloc((void**)&deviceMesh->verts, n * sizeof(int));
    cudaErrCheck(cuda_ret, "Unable to allocate device memory for vert array");
	cuda_ret = cudaMalloc((void**)&deviceMesh->edges, n * sizeof(int));
    cudaErrCheck(cuda_ret, "Unable to allocate device memory for edge array");

    if(isQuad == 0) {
        //only allocate enough for the very first mesh
        cuda_ret = cudaMalloc((void**)&deviceMesh->nexts, n0 * sizeof(int));
        cudaErrCheck(cuda_ret, "Unable to allocate device memory for next array");
        cuda_ret = cudaMalloc((void**)&deviceMesh->prevs, n0 * sizeof(int));
        cudaErrCheck(cuda_ret, "Unable to allocate device memory for prev array");
        cuda_ret = cudaMalloc((void**)&deviceMesh->faces, n0 * sizeof(int));
        cudaErrCheck(cuda_ret, "Unable to allocate device memory for face array");
    }
    printf("Device memory allocation completed\n\n");
}

void copyHostToDeviceMesh(Mesh* from, DeviceMesh* to) {
    cudaError_t cuda_ret;
	printf("Copying mesh from host to device...\n"); 

    int m = from->numVerts;
	cuda_ret = cudaMemcpy(to->xCoords, from->xCoords, m * sizeof(float), cudaMemcpyHostToDevice);
    cudaErrCheck(cuda_ret, "Unable to copy x-coordinates to the device");
	cuda_ret = cudaMemcpy(to->yCoords, from->yCoords, m * sizeof(float), cudaMemcpyHostToDevice);
    cudaErrCheck(cuda_ret, "Unable to copy y-coordinates to the device")
	cuda_ret = cudaMemcpy(to->zCoords, from->zCoords, m * sizeof(float), cudaMemcpyHostToDevice);
    cudaErrCheck(cuda_ret, "Unable to copy z-coordinates to the device");

	int n = from->numHalfEdges;
	cuda_ret = cudaMemcpy(to->twins, from->twins, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaErrCheck(cuda_ret, "Unable to copy twins to the device");
	cuda_ret = cudaMemcpy(to->verts, from->verts, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaErrCheck(cuda_ret, "Unable to copy verts to the device");
	cuda_ret = cudaMemcpy(to->edges, from->edges, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaErrCheck(cuda_ret, "Unable to copy edges to the device");
    
    if(from->isQuad == 0) {
        cuda_ret = cudaMemcpy(to->nexts, from->nexts, n * sizeof(int), cudaMemcpyHostToDevice);
        cudaErrCheck(cuda_ret, "Unable to copy nexts to the device");
        cuda_ret = cudaMemcpy(to->prevs, from->prevs, n * sizeof(int), cudaMemcpyHostToDevice);
        cudaErrCheck(cuda_ret, "Unable to copy prevs to the device"); 
        cuda_ret = cudaMemcpy(to->faces, from->faces, n * sizeof(int), cudaMemcpyHostToDevice);
        cudaErrCheck(cuda_ret, "Unable to copy faces to the device");
    }

	printf("Copy to device completed\n\n");
}

Mesh copyDeviceMeshToHostMesh(DeviceMesh* from) {
    cudaError_t cuda_ret;
    
	printf("Copying mesh from device back to host...\n");

    Mesh to = initMesh(from->numVerts, from->numHalfEdges, from->numFaces, from->numEdges);
    to.isQuad = 1;
    allocQuadMesh(&to);

    // to already has the correct values for numVerts, numFaces etc.
    int m = to.numVerts;
	cuda_ret = cudaMemcpy(to.xCoords, from->xCoords, m * sizeof(float), cudaMemcpyDeviceToHost);
    cudaErrCheck(cuda_ret, "Unable to copy x-coordinates from the device");
	cuda_ret = cudaMemcpy(to.yCoords, from->yCoords, m * sizeof(float), cudaMemcpyDeviceToHost);
    cudaErrCheck(cuda_ret, "Unable to copy y-coordinates from the device")
	cuda_ret = cudaMemcpy(to.zCoords, from->zCoords, m * sizeof(float), cudaMemcpyDeviceToHost);
    cudaErrCheck(cuda_ret, "Unable to copy z-coordinates from the device");

	int n = to.numHalfEdges;
	cuda_ret = cudaMemcpy(to.twins, from->twins, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaErrCheck(cuda_ret, "Unable to copy twins from the device");
	cuda_ret = cudaMemcpy(to.verts, from->verts, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaErrCheck(cuda_ret, "Unable to copy verts from the device");
	cuda_ret = cudaMemcpy(to.edges, from->edges, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaErrCheck(cuda_ret, "Unable to copy edges from the device");
    return to;
}