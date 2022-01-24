#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "cudaSubdivision.cuh"
#include "kernelInvoker.cuh"
#include "math.h"
#include "mesh/deviceMesh.cuh"
#include "util/deviceCommunication.cuh"
#include "util/util.cuh"

/**
 * @brief Get the Number Of Verts At Level object
 * 
 * @param d 
 * @param v1 
 * @param e1 
 * @param f1 
 * @return int 
 */
int getNumberOfVertsAtLevel(int d, int v1, int e1, int f1) {
	int finalNumberOfVerts = v1;
    if (d > 1) {
        finalNumberOfVerts += pow(2, d - 1) * (e1 + (pow(2, d) - 1) * f1);
    }
	return finalNumberOfVerts;
}

/**
 * @brief 
 * 
 * @param in 
 * @param out 
 * @param mesh 
 * @param subdivisionLevel 
 */
void allocateDeviceMemoryMeshes(DeviceMesh* in, DeviceMesh* out, Mesh* mesh, int subdivisionLevel) {
    int v1 = mesh->numVerts + mesh->numFaces + mesh->numEdges;
    int e1 = 2 * mesh->numEdges + mesh->numHalfEdges;
    int f1 = mesh->numHalfEdges;

	int bufferSizeAVerts;
	int bufferSizeAHalfEdges;
	int bufferSizeBVerts;
	int bufferSizeBHalfEdges;
	
	bufferSizeAVerts = getNumberOfVertsAtLevel(subdivisionLevel - 1, v1, e1, f1);
	bufferSizeAHalfEdges = pow(4, subdivisionLevel - 1) * mesh->numHalfEdges;
	bufferSizeBVerts = getNumberOfVertsAtLevel(subdivisionLevel, v1, e1, f1);
	bufferSizeBHalfEdges = bufferSizeAHalfEdges * 4;

	if(subdivisionLevel % 2 == 0) {
		// if odd, then buffer A is largest
		swap(&bufferSizeAVerts, &bufferSizeBVerts);
		swap(&bufferSizeAHalfEdges, &bufferSizeBHalfEdges);
	}

    printf("\tAllocating device variables...\n");
    if (mesh->isQuad) {
        allocateDeviceMemoryQuad(in, bufferSizeAVerts, bufferSizeAHalfEdges);
    } else {
        allocateDeviceMemory(in, bufferSizeAVerts, bufferSizeAHalfEdges, mesh->numHalfEdges);
    }
    allocateDeviceMemoryQuad(out, bufferSizeBVerts, bufferSizeBHalfEdges);
    printf("\tDevice memory allocation completed\n");
}

/**
 * @brief 
 * 
 * @param mesh 
 * @param subdivisionLevel 
 * @return Mesh 
 */
Mesh cudaSubdivide(Mesh* mesh, int subdivisionLevel) {
    cudaError_t cuda_ret;
    printf("Starting Subdvision process...\n\n");

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

    printf("\nSubdivision Process Complete!\n");
    return result_h;
}