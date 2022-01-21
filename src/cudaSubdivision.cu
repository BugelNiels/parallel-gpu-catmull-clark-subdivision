#include "cudaSubdivision.cuh"

#include <stdlib.h>
#include <stdio.h>
#include "math.h"
#include <assert.h>

#include "util/util.cuh"
#include "kernelInvoker.cuh"
#include "util/deviceCommunication.cuh"
#include "mesh/deviceMesh.cuh"

void allocateDeviceMemoryMeshes(DeviceMesh* in, DeviceMesh* out, Mesh* mesh, int subdivisionLevel) {
	int finalNumberOfHalfEdges = pow(4, subdivisionLevel) * mesh->numHalfEdges;
	int v1 = mesh->numVerts + mesh->numFaces + mesh->numEdges;
	int e1 = 2 * mesh->numEdges + mesh->numHalfEdges;
	int f1 = mesh->numHalfEdges;
	int finalNumberOfVerts = v1;
	if(subdivisionLevel > 1) {
		finalNumberOfVerts += pow(2, subdivisionLevel - 1) * (e1 + (pow(2, subdivisionLevel) -1) * f1);
	}

	// Future TODO: in mesh does not need as much memory only D-1; which mesh get these, depends on the number of subdivision levels
	allocateDeviceMemory(in, finalNumberOfVerts, finalNumberOfHalfEdges, mesh->numHalfEdges, mesh->isQuad);
	allocateDeviceMemory(out, finalNumberOfVerts, finalNumberOfHalfEdges, 0, 0);
}

Mesh cudaSubdivide(Mesh* mesh, int subdivisionLevel) {
	cudaError_t cuda_ret;
	printf("Starting Subdvision\n");
	
	// use double buffering; calculate final number of half edges and numVerts and allocat out and in arrays
	// switch each subdivision level
	DeviceMesh in = createEmptyCopyOnDevice(mesh);
	DeviceMesh out = createEmptyCopyOnDevice(mesh);

	allocateDeviceMemoryMeshes(&in, &out, mesh, subdivisionLevel);

	cuda_ret = cudaDeviceSynchronize();
	cudaErrCheck(cuda_ret, "Unable to sync");

	DeviceMesh result_d = performSubdivision(&in, &out, subdivisionLevel, mesh);

	Mesh result_h = copyDeviceMeshToHostMesh(&result_d);
	
	cuda_ret = cudaDeviceSynchronize();
	cudaErrCheck(cuda_ret, "Unable to sync");

	freeDeviceMesh(&in);
	freeDeviceMesh(&out);

	printf("Subdivision Complete!\n");
	return result_h;
}