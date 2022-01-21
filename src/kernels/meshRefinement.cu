#include "quadRefinement.cuh"
#include "../util/util.cuh"
#include "kernelUtil/kernelUtils.cuh"

__global__ void resetMesh(DeviceMesh* in, DeviceMesh* out) {
    
    int numVerts = in->numVerts + in->numFaces + in->numEdges;
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int v = i; v < numVerts; v += stride) {
        out->xCoords[v] = 0;
        out->yCoords[v] = 0;
        out->zCoords[v] = 0;
    } 
    
    if(blockIdx.x == 0 && threadIdx.x == 0) {
        int h = in->numHalfEdges;
        out->numEdges = 2 * in->numEdges + h;
        out->numFaces = h;
        out->numHalfEdges = h * 4;
        out->numVerts = numVerts;
    }
}
__device__ void topologyRefinement(int h, DeviceMesh* in, DeviceMesh* out, int vd, int fd, int ed) {
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

__device__ void facePoint(int h, DeviceMesh* in, DeviceMesh* out) {
    int v = in->verts[h];
    int i = in->numVerts + in->faces[h];
    float m = (float)cycleLength(h, in);
    atomicAdd(&out->xCoords[i], in->xCoords[v] / m);
    atomicAdd(&out->yCoords[i], in->yCoords[v] / m);
    atomicAdd(&out->zCoords[i], in->zCoords[v] / m);
}

__device__ void edgePoint(int h, DeviceMesh* in, DeviceMesh* out) {
    int vd = in->numVerts;
    int fd = in->numFaces;
    int v = in->verts[h];
    int j = vd + fd + in->edges[h];
    float x, y, z;
    // boundary
    if(in->twins[h] < 0) {
        int i = in->verts[in->nexts[h]];
        x = (in->xCoords[v] + in->xCoords[i]) / 2.0f;
        y = (in->yCoords[v] + in->yCoords[i]) / 2.0f;
        z = (in->zCoords[v] + in->zCoords[i]) / 2.0f;      
    } else {
        int i = vd + in->faces[h];
        x = (in->xCoords[v] + out->xCoords[i]) / 4.0f;
        y = (in->yCoords[v] + out->yCoords[i]) / 4.0f;
        z = (in->zCoords[v] + out->zCoords[i]) / 4.0f;
    }    
    atomicAdd(&out->xCoords[j], x);
    atomicAdd(&out->yCoords[j], y);
    atomicAdd(&out->zCoords[j], z);
}

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

    int k = in->verts[in->nexts[h]];
    // do similar thing for the next vertex
    x = (edgex + in->xCoords[k]) / 4.0f;
    y = (edgey + in->yCoords[k]) / 4.0f;
    z = (edgez + in->zCoords[k]) / 4.0f;
    atomicAdd(&out->xCoords[k], x);
    atomicAdd(&out->yCoords[k], y);
    atomicAdd(&out->zCoords[k], z);
}

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

__global__ void refineTopology(DeviceMesh* in, DeviceMesh* out) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    int vd = in->numVerts;
    int fd = in->numFaces;
    int ed = in->numEdges;
    int hd = in->numHalfEdges;
    for(int i = h; i < hd; i += stride) {
        topologyRefinement(i, in, out, vd, fd, ed);
    }    
}

__global__ void facePoints(DeviceMesh* in, DeviceMesh* out) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int hd = in->numHalfEdges;
    for(int i = h; i < hd; i += stride) {
        facePoint(i, in, out);
    } 
}

__global__ void edgePoints(DeviceMesh* in, DeviceMesh* out) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int hd = in->numHalfEdges;
    for(int i = h; i < hd; i += stride) {
        edgePoint(i, in, out);
    } 
}

__global__ void vertexPoints(DeviceMesh* in, DeviceMesh* out) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int hd = in->numHalfEdges;
    for(int i = h; i < hd; i += stride) {        
        float n = valence(i, in);
        if(n > 0) {
            vertexPoint(i, in, out, n);
        } else if(in->twins[i] < 0) {
            boundaryVertexPoint(i, in, out);
        }
    }   
}