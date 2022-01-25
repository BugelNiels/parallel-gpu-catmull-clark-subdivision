#include "kernelInvoker.cuh"
#include "kernels/meshRefinement.cuh"
#include "kernels/optimisedQuadRef.cuh"
#include "kernels/quadRefinement.cuh"
#include "math.h"
#include "stdio.h"
#include "util/deviceCommunication.cuh"
#include "util/util.cuh"

// Set to 0 to use the regular kernels for quad meshes
// Set to 1 for the optimized single kernel for quad meshes
#define USE_OPTIMIZED_KERNEL 1

/**
 * @brief Swaps the pointers to two device meshes
 *
 * @param prevMeshPtr Pointer to the first mesh
 * @param newMeshPtr Pointer to the second mesh
 */
void meshSwap(DeviceMesh** prevMeshPtr, DeviceMesh** newMeshPtr) {
    DeviceMesh* temp = *prevMeshPtr;
    *prevMeshPtr = *newMeshPtr;
    *newMeshPtr = temp;
}

/**
 * @brief Performs Catmull-Clark subdivision up to subdivisionLevel on the GPU. Uses double-buffering
 *
 * @param input Device Mesh buffer A
 * @param output Device Mesh buffer B
 * @param subdivisionLevel The subdivision level to subdivide to
 * @param mesh The host mesh to subdivide
 * @return DeviceMesh
 */
DeviceMesh performSubdivision(DeviceMesh* input, DeviceMesh* output, int subdivisionLevel, Mesh* mesh) {
    cudaError_t cuda_ret;
    cudaEvent_t start, stop;

    int h0 = mesh->numHalfEdges;
    int v0 = mesh->numVerts;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Convert only the struct to a pointer on the device
    DeviceMesh* in = toDevicePointer(input);
    DeviceMesh* out = toDevicePointer(output);

    dim3 dim_grid, dim_block;
    dim_block.x = BLOCK_SIZE;
    dim_block.y = dim_block.z = 1;
    dim_grid.y = dim_grid.z = 1;

    printf("\n\t------------------\n\tStarting subdivision kernels...\n");

    // device must be synced before this point
    // all the stuff before this can be pre-allocated/pre-calculated
    cudaEventRecord(start);
    copyHostToDeviceMesh(mesh, input);

    int startLevel = 0;
    int v = v0;
    if (!mesh->isQuad) {
        // First subdivision step should not use the quad kernels for non-quad meshes
        // After the first subdivision step, the mesh consists of only quads and then those kernels can be used safely
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
        int he = pow(4, startLevel) * h0;
        dim_grid.x = MIN((he - 1) / BLOCK_SIZE + 1, MAX_GRID_SIZE);
        // flip sign of boundary vertices
        setBoundaryVerts<<<dim_grid, dim_block>>>(in);
    }

    for (int d = startLevel; d < subdivisionLevel; d++) {
        // each thread covers 1 half edge. Number of half edges can be much
        // greater than blockdim * gridDim, so limit to MAX_GRID_SIZE.
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

    // Output to timings.txt
    FILE* timingsFile = fopen("timings.txt", "a");
    fprintf(timingsFile, "%lf\n", milliseconds);

    DeviceMesh m = devicePointerToHostMesh(in);
    cudaFree(in);
    cudaFree(out);
    return m;
}