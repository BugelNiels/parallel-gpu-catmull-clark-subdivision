#include "kernelInvoker.cuh"
#include "kernels/meshRefinement.cuh"
#include "kernels/optimisedQuadRef.cuh"
#include "kernels/quadRefinement.cuh"
#include "math.h"
#include "stdio.h"
#include "util/deviceCommunication.cuh"
#include "util/util.cuh"

#define USE_OPTIMIZED_KERNEL 1

// swaps pointers
void meshSwap(DeviceMesh** prevMeshPtr, DeviceMesh** newMeshPtr) {
    DeviceMesh* temp = *prevMeshPtr;
    *prevMeshPtr = *newMeshPtr;
    *newMeshPtr = temp;
}

DeviceMesh performSubdivision(DeviceMesh* input, DeviceMesh* output, int subdivisionLevel, Mesh* mesh) {
    cudaError_t cuda_ret;
    cudaEvent_t start, stop;

    int h0 = mesh->numHalfEdges;
    int v0 = mesh->numVerts;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    DeviceMesh* in = toDevicePointer(input);
    DeviceMesh* out = toDevicePointer(output);

    dim3 dim_grid, dim_block;

    // each thread takes 1 half edge
    dim_block.x = BLOCK_SIZE;
    dim_block.y = dim_block.z = 1;
    dim_grid.y = dim_grid.z = 1;

    printf("\n\t------------------\n\tStarting subdivision kernels...\n");

    // device must be synced before this point
    cudaEventRecord(start);
    // all the stuff before this can be pre-allocated/pre-calculated

    copyHostToDeviceMesh(mesh, input);

    int startLevel = 0;
    int v = v0;
    if (!mesh->isQuad) {
        dim_grid.x = MIN((h0 - 1) / BLOCK_SIZE + 1, MAX_GRID_SIZE);
        resetMesh<<<dim_grid, dim_block>>>(in, out);
        refineTopology<<<dim_grid, dim_block>>>(in, out);
        facePoints<<<dim_grid, dim_block>>>(in, out);
        edgePoints<<<dim_grid, dim_block>>>(in, out);
        vertexPoints<<<dim_grid, dim_block>>>(in, out);
        meshSwap(&in, &out);
        startLevel = 1;
        v = mesh->numVerts + mesh->numFaces + mesh->numEdges;
    }
    if (USE_OPTIMIZED_KERNEL) {
        // flip sign of boundary vertices
        int he = pow(4, startLevel) * h0;
        dim_grid.x = MIN((he - 1) / BLOCK_SIZE + 1, MAX_GRID_SIZE);
        setBoundaryVerts<<<dim_grid, dim_block>>>(in);
    }

    for (int d = startLevel; d < subdivisionLevel; d++) {
        // each thread covers 1 half edge. Number of half edges can be much
        // greater than blockdim * gridDim.
        int he = pow(4, d) * h0;
        dim_grid.x = MIN((he - 1) / BLOCK_SIZE + 1, MAX_GRID_SIZE);
        resetMesh<<<dim_grid, dim_block>>>(in, out);
        if (USE_OPTIMIZED_KERNEL) {
            optimisedSubdivide<<<dim_grid, dim_block>>>(in, out, v);
        } else {
            quadRefineTopology<<<dim_grid, dim_block>>>(in, out);
            quadFacePoints<<<dim_grid, dim_block>>>(in, out);
            quadEdgePoints<<<dim_grid, dim_block>>>(in, out);
            quadVertexPoints<<<dim_grid, dim_block>>>(in, out);
        }
        // result is in "out"; after this swap, the result is in "in"
        meshSwap(&in, &out);
    }
    cudaEventRecord(stop);
    cuda_ret = cudaDeviceSynchronize();
    cudaErrCheck(cuda_ret, "Unable to execute kernel");

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("\tSubdivision kernels completed\n\tExecution took: %lf msec\n\t------------------\n\n", milliseconds);
    FILE* timingsFile = fopen("timings.txt", "a");
    fprintf(timingsFile, "%lf\n", milliseconds);
    DeviceMesh m = devicePointerToHostMesh(in);

    cudaFree(in);
    cudaFree(out);
    return m;
}