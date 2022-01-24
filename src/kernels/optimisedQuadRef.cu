#include "../util/util.cuh"
#include "kernelUtil/kernelUtils.cuh"
#include "math.h"
#include "quadRefinement.cuh"
#include "stdio.h"

/**
 * @brief Flips the sign of VERT(h) when VERT(h) is a boundary vertex
 *
 * @param in Half-edge mesh at level d
 */
__global__ void setBoundaryVerts(DeviceMesh* in) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int hd = in->numHalfEdges;
    for (int i = h; i < hd; i += stride) {
        if (valenceQuad(i, in) < 0) {
            in->verts[i] *= -1;
        }
    }
}

/**
 * @brief Single optimized refinement kernel. Does both the topology and geometry refinement in a single kernel.
 * Does some fancy tricks do reduce the runtime.
 *
 * @param in Half-edge mesh at level d
 * @param out Half-edge mesh at level d+1
 * @param v0 Number of vertices at level 1 (level 0 in case of quad meshes)
 */
__global__ void optimisedSubdivide(DeviceMesh* in, DeviceMesh* out, int v0) {
    /**
     * Store face points in shared memory. Face points are frequently accessed. This is possible since faces are stored
     * contiguously in memory and the BLOCK_SIZE is a multiple of 4
     */
    __shared__ float facePointsX[FACES_PER_BLOCK];
    __shared__ float facePointsY[FACES_PER_BLOCK];
    __shared__ float facePointsZ[FACES_PER_BLOCK];

    int vd = in->numVerts;
    int fd = in->numFaces;
    int ed = in->numEdges;

    int ti = threadIdx.x / 4;
    int t2 = threadIdx.x % 4;

    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int h = start; h < in->numHalfEdges; h += stride) {
        // Threads are scheduled in warps. To prevent one warp from resetting everything, while another has not finished
        // its calculations yet, only reset its own face points. This works because WARP_SIZE is a multiple of 4. This
        // eliminates the need for a threadsync
        if (t2 == 0) {
            // reset shared memory
            facePointsX[ti] = 0;
            facePointsY[ti] = 0;
            facePointsZ[ti] = 0;
        }
        // sign of vBound can be used for efficient boundary testing
        int vBound = in->verts[h];
        int v = abs(vBound);
        int hp = prev(h);
        int he = in->edges[h];
        int ht = in->twins[h];

        // topology refinement
        out->twins[4 * h] = ht < 0 ? -1 : 4 * next(ht) + 3;
        out->twins[4 * h + 1] = 4 * next(h) + 2;
        out->twins[4 * h + 2] = 4 * hp + 1;
        out->twins[4 * h + 3] = 4 * in->twins[hp];

        out->verts[4 * h] = v;
        out->verts[4 * h + 1] = vBound >= 0 ? vd + fd + he : -vd - fd - he;
        out->verts[4 * h + 2] = vd + face(h);
        out->verts[4 * h + 3] = vd + fd + in->edges[hp];

        out->edges[4 * h] = h > ht ? 2 * he : 2 * he + 1;
        out->edges[4 * h + 1] = 2 * ed + h;
        out->edges[4 * h + 2] = 2 * ed + hp;
        out->edges[4 * h + 3] = hp > in->twins[hp] ? 2 * in->edges[hp] + 1 : 2 * in->edges[hp];

        // Face Points
        float invX = in->xCoords[v];
        float invY = in->yCoords[v];
        float invZ = in->zCoords[v];
        atomicAdd(&facePointsX[ti], invX / 4.0f);
        atomicAdd(&facePointsY[ti], invY / 4.0f);
        atomicAdd(&facePointsZ[ti], invZ / 4.0f);

        // Edge Points
        // Always calculate the edge point produced by boundary edge rule. Used so that vertex point calculation can be
        // done within this single kernel
        int vNext = abs(in->verts[next(h)]);
        float edgex = (invX + in->xCoords[vNext]) / 2.0f;
        float edgey = (invY + in->yCoords[vNext]) / 2.0f;
        float edgez = (invZ + in->zCoords[vNext]) / 2.0f;

        float x, y, z;
        if (ht >= 0) {
            // average the vertex of this vertex and the face point
            x = (invX + facePointsX[ti]) / 4.0f;
            y = (invY + facePointsY[ti]) / 4.0f;
            z = (invZ + facePointsZ[ti]) / 4.0f;
        } else {
            // boundary edge point
            x = edgex;
            y = edgey;
            z = edgez;
        }
        int j = vd + fd + he;
        atomicAdd(&out->xCoords[j], x);
        atomicAdd(&out->yCoords[j], y);
        atomicAdd(&out->zCoords[j], z);

        // Vertex Points
        if (vBound >= 0) {
            // newly added interior face and edge points always have valence 4
            float n = v >= v0 ? 4 : valenceQuad(h, in);
            float n2 = n * n;
            x = (2 * edgex + facePointsX[ti] + (n - 3) * invX) / n2;
            y = (2 * edgey + facePointsY[ti] + (n - 3) * invY) / n2;
            z = (2 * edgez + facePointsZ[ti] + (n - 3) * invZ) / n2;
            atomicAdd(&out->xCoords[v], x);
            atomicAdd(&out->yCoords[v], y);
            atomicAdd(&out->zCoords[v], z);
        } else if (ht < 0) {
            // boundary vertex point
            // only needs an update from its boundary half-edge. Interior half-edges of boundary vertices do nothing
            x = (edgex + invX) / 4.0f;
            y = (edgey + invY) / 4.0f;
            z = (edgez + invZ) / 4.0f;
            atomicAdd(&out->xCoords[v], x);
            atomicAdd(&out->yCoords[v], y);
            atomicAdd(&out->zCoords[v], z);

            // Compute contribution of this edge point to VERT(NEXT(h))
            x = (edgex + in->xCoords[vNext]) / 4.0f;
            y = (edgey + in->yCoords[vNext]) / 4.0f;
            z = (edgez + in->zCoords[vNext]) / 4.0f;
            atomicAdd(&out->xCoords[vNext], x);
            atomicAdd(&out->yCoords[vNext], y);
            atomicAdd(&out->zCoords[vNext], z);
        }
        // Put face points in the half-edge mesh at level d+1
        if (t2 == 0) {
            int ind = vd + face(h);
            out->xCoords[ind] = facePointsX[ti];
            out->yCoords[ind] = facePointsY[ti];
            out->zCoords[ind] = facePointsZ[ti];
        }
    }
}
