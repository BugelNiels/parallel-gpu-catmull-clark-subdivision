#include "../util/util.cuh"
#include "kernelUtil/kernelUtils.cuh"
#include "quadRefinement.cuh"

/**
 * @brief Resets the vertex coordinates of the mesh and calculates the new number of edges, faces, half-edges and
 * vertices at level d+1
 *
 * @param in Mesh at level
 * @param out Mesh to reset.
 */
__global__ void resetMesh(DeviceMesh* in, DeviceMesh* out) {
    int numVerts = in->numVerts + in->numFaces + in->numEdges;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int v = i; v < numVerts; v += stride) {
        out->xCoords[v] = 0;
        out->yCoords[v] = 0;
        out->zCoords[v] = 0;
    }

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int h = in->numHalfEdges;
        out->numEdges = 2 * in->numEdges + h;
        out->numFaces = h;
        out->numHalfEdges = h * 4;
        out->numVerts = numVerts;
    }
}

/**
 * @brief Topology refinement of a single half-edge. Sets the properties of the 4 half-edges generated from the provided
 * half-edge.
 *
 * @param h Half-edge index at level d
 * @param in Half-edge mesh at level d
 * @param out Half-edge mesh at level d+1
 * @param vd Number of vertices at level d
 * @param fd Number of faces at level d
 * @param ed Number of edges at level d
 */
__device__ void refineEdge(int h, DeviceMesh* in, DeviceMesh* out, int vd, int fd, int ed) {
    int hp = in->prevs[h];
    int he = in->edges[h];

    int ht = in->twins[h];
    int thp = in->twins[hp];
    int ehp = in->edges[hp];

    out->twins[4 * h] = ht < 0 ? -1 : 4 * in->nexts[ht] + 3;
    out->twins[4 * h + 1] = 4 * in->nexts[h] + 2;
    out->twins[4 * h + 2] = 4 * hp + 1;
    out->twins[4 * h + 3] = 4 * thp;

    out->verts[4 * h] = in->verts[h];
    out->verts[4 * h + 1] = vd + fd + he;
    out->verts[4 * h + 2] = vd + in->faces[h];
    out->verts[4 * h + 3] = vd + fd + ehp;

    out->edges[4 * h] = h > ht ? 2 * he : 2 * he + 1;
    out->edges[4 * h + 1] = 2 * ed + h;
    out->edges[4 * h + 2] = 2 * ed + hp;
    out->edges[4 * h + 3] = hp > thp ? 2 * ehp + 1 : 2 * ehp;
}

/**
 * @brief Refines the topology for the half-edges.
 *
 * @param in Half-edge mesh at level d
 * @param out Half-edge mesh at level d+1
 */
__global__ void refineTopology(DeviceMesh* in, DeviceMesh* out) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int vd = in->numVerts;
    int fd = in->numFaces;
    int ed = in->numEdges;
    int hd = in->numHalfEdges;
    for (int i = h; i < hd; i += stride) {
        refineEdge(i, in, out, vd, fd, ed);
    }
}

/**
 * @brief Calculates the contribution of the provided half-edge to its face point.
 *
 * @param h Half-edge index at level d
 * @param in Half-edge mesh at level d
 * @param out Half-edge mesh at level d+1
 */
__device__ void facePoint(int h, DeviceMesh* in, DeviceMesh* out) {
    int v = in->verts[h];
    int i = in->numVerts + in->faces[h];
    float m = (float)cycleLength(h, in);
    atomicAdd(&out->xCoords[i], in->xCoords[v] / m);
    atomicAdd(&out->yCoords[i], in->yCoords[v] / m);
    atomicAdd(&out->zCoords[i], in->zCoords[v] / m);
}

/**
 * @brief Calculates the contribution of the provided half-edge to its edge point
 *
 * @param h Half-edge index at level d
 * @param in Half-edge mesh at level d
 * @param out Half-edge mesh at level d+1
 */
__device__ void edgePoint(int h, DeviceMesh* in, DeviceMesh* out) {
    int vd = in->numVerts;
    int fd = in->numFaces;
    int v = in->verts[h];
    int j = vd + fd + in->edges[h];
    if (in->twins[h] >= 0) {
        int i = vd + in->faces[h];
        float x = (in->xCoords[v] + out->xCoords[i]) / 4.0f;
        float y = (in->yCoords[v] + out->yCoords[i]) / 4.0f;
        float z = (in->zCoords[v] + out->zCoords[i]) / 4.0f;
        atomicAdd(&out->xCoords[j], x);
        atomicAdd(&out->yCoords[j], y);
        atomicAdd(&out->zCoords[j], z);
    } else {
        // boundary
        int vNext = in->verts[next(h)];
        out->xCoords[j] = (in->xCoords[v] + in->xCoords[vNext]) / 2.0f;
        out->yCoords[j] = (in->yCoords[v] + in->yCoords[vNext]) / 2.0f;
        out->zCoords[j] = (in->zCoords[v] + in->zCoords[vNext]) / 2.0f;
    }
}

/**
 * @brief Calculates the contribution of this half-edge to its vertex point and the vertex point of NEXT(h)
 *
 * @param h Half-edge index at level d
 * @param in Half-edge mesh at level d
 * @param out Half-edge mesh at level d+1
 */
__device__ void boundaryVertexPoint(int h, DeviceMesh* in, DeviceMesh* out) {
    int v = in->verts[h];
    int vd = in->numVerts;
    int j = vd + in->numFaces + in->edges[h];
    float edgex = out->xCoords[j];
    float edgey = out->yCoords[j];
    float edgez = out->zCoords[j];

    float x = (edgex + in->xCoords[v]) / 4.0f;
    float y = (edgey + in->yCoords[v]) / 4.0f;
    float z = (edgez + in->zCoords[v]) / 4.0f;
    atomicAdd(&out->xCoords[v], x);
    atomicAdd(&out->yCoords[v], y);
    atomicAdd(&out->zCoords[v], z);

    int vNext = in->verts[in->nexts[h]];
    // do similar thing for the next vertex
    x = (edgex + in->xCoords[vNext]) / 4.0f;
    y = (edgey + in->yCoords[vNext]) / 4.0f;
    z = (edgez + in->zCoords[vNext]) / 4.0f;
    atomicAdd(&out->xCoords[vNext], x);
    atomicAdd(&out->yCoords[vNext], y);
    atomicAdd(&out->zCoords[vNext], z);
}

/**
 * @brief Calculates the contribution of the provided half-edge to the vertex point
 *
 * @param h Half-edge index at level d
 * @param in Half-edge mesh at level d
 * @param out Half-edge mesh at level d+1
 * @param n Valence of the vertex
 */
__device__ void vertexPoint(int h, DeviceMesh* in, DeviceMesh* out, int n) {
    int v = in->verts[h];

    int vd = in->numVerts;
    int i = vd + in->faces[h];
    int j = vd + in->numFaces + in->edges[h];
    float n2 = n * n;
    float x = (4 * out->xCoords[j] - out->xCoords[i] + (n - 3) * in->xCoords[v]) / n2;
    float y = (4 * out->yCoords[j] - out->yCoords[i] + (n - 3) * in->yCoords[v]) / n2;
    float z = (4 * out->zCoords[j] - out->zCoords[i] + (n - 3) * in->zCoords[v]) / n2;
    atomicAdd(&out->xCoords[v], x);
    atomicAdd(&out->yCoords[v], y);
    atomicAdd(&out->zCoords[v], z);
}

/**
 * @brief Calculates the positions of all face points
 *
 * @param in Half-edge mesh at level d
 * @param out Half-edge mesh at level d+1
 */
__global__ void facePoints(DeviceMesh* in, DeviceMesh* out) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int hd = in->numHalfEdges;
    for (int i = h; i < hd; i += stride) {
        facePoint(i, in, out);
    }
}

/**
 * @brief Calculates the positions of all edge points
 *
 * @param in Half-edge mesh at level d
 * @param out Half-edge mesh at level d+1
 */
__global__ void edgePoints(DeviceMesh* in, DeviceMesh* out) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int hd = in->numHalfEdges;
    for (int i = h; i < hd; i += stride) {
        edgePoint(i, in, out);
    }
}

/**
 * @brief Calculates the positions of all vertex points
 *
 * @param in Half-edge mesh at level d
 * @param out Half-edge mesh at level d+1
 */
__global__ void vertexPoints(DeviceMesh* in, DeviceMesh* out) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int hd = in->numHalfEdges;
    for (int i = h; i < hd; i += stride) {
        float n = valence(i, in);
        if (n > 0) {
            vertexPoint(i, in, out, n);
        } else if (in->twins[i] < 0) {
            boundaryVertexPoint(i, in, out);
        }
    }
}