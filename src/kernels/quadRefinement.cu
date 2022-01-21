#include "quadRefinement.cuh"
#include "../util/util.cuh"
#include "kernelUtil/kernelUtils.cuh"

__device__ void quadRefineEdges(int h, DeviceMesh* in, DeviceMesh* out, int vd, int fd, int ed) {
    int hp = prev(h);
    int he = in->edges[h];

    int ht = in->twins[h];
    int thp = in->twins[hp];
    int ehp = in->edges[hp];
    
    out->twins[4 * h] = ht < 0 ? -1 : 4 * next(ht) + 3;
    out->twins[4 * h + 1] = 4 * next(h) + 2;
    out->twins[4 * h + 2] = 4 * hp + 1;
    out->twins[4 * h + 3] = 4 * thp;

    out->verts[4 * h] = in->verts[h];
    out->verts[4 * h + 1] = vd + fd + he;
    out->verts[4 * h + 2] = vd + face(h);
    out->verts[4 * h + 3] = vd + fd + ehp;

    out->edges[4 * h] = h > ht ? 2 * he : 2 * he + 1;
    out->edges[4 * h + 1] = 2 * ed + h;
    out->edges[4 * h + 2] = 2 * ed + hp;
    out->edges[4 * h + 3] = hp > thp ? 2 * ehp + 1 : 2 * ehp;
}

__device__ void quadFacePoint(int h, DeviceMesh* in, DeviceMesh* out) {
    int v = in->verts[h];
    int i = in->numVerts + face(h);
    atomicAdd(&out->xCoords[i], in->xCoords[v] / 4.0f);
    atomicAdd(&out->yCoords[i], in->yCoords[v] / 4.0f);
    atomicAdd(&out->zCoords[i], in->zCoords[v] / 4.0f);
}

__device__ void quadEdgePoint(int h, DeviceMesh* in, DeviceMesh* out) {
    int vd = in->numVerts;
    int fd = in->numFaces;
    int v = in->verts[h];
    int j = vd + fd + in->edges[h];

    if(in->twins[h] >= 0) {
        int i = vd + face(h);
        float x = (in->xCoords[v] + out->xCoords[i]) / 4.0f;
        float y = (in->yCoords[v] + out->yCoords[i]) / 4.0f;
        float z = (in->zCoords[v] + out->zCoords[i]) / 4.0f;
        atomicAdd(&out->xCoords[j], x);
        atomicAdd(&out->yCoords[j], y);
        atomicAdd(&out->zCoords[j], z);
    } else  {
        // boundary
        int i = in->verts[next(h)];
        out->xCoords[j] = (in->xCoords[v] + in->xCoords[i]) / 2.0f;
        out->yCoords[j] = (in->yCoords[v] + in->yCoords[i]) / 2.0f;
        out->zCoords[j] = (in->zCoords[v] + in->zCoords[i]) / 2.0f;   
    }   
}


__device__ void quadBoundaryVertexPoint(int h, DeviceMesh* in, DeviceMesh* out) {
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

    int k = in->verts[next(h)];
    // do similar thing for the next vertex
    x = (edgex + in->xCoords[k]) / 4.0f;
    y = (edgey + in->yCoords[k]) / 4.0f;
    z = (edgez + in->zCoords[k]) / 4.0f;
    atomicAdd(&out->xCoords[k], x);
    atomicAdd(&out->yCoords[k], y);
    atomicAdd(&out->zCoords[k], z);
}

__device__ void quadVertexPoint(int h, DeviceMesh* in, DeviceMesh* out, int n) {
    int v = in->verts[h];
    int vd = in->numVerts;
    int i = vd + face(h);
    int j = vd + in->numFaces + in->edges[h];
    float n2 = n * n;
    float x = (4 * out->xCoords[j] - out->xCoords[i] + (n - 3) * in->xCoords[v]) / n2;
    float y = (4 * out->yCoords[j] - out->yCoords[i] + (n - 3) * in->yCoords[v]) / n2;
    float z = (4 * out->zCoords[j] - out->zCoords[i] + (n - 3) * in->zCoords[v]) / n2;

    atomicAdd(&out->xCoords[v], x);
    atomicAdd(&out->yCoords[v], y);
    atomicAdd(&out->zCoords[v], z);
}

__global__ void quadRefineTopology(DeviceMesh* in, DeviceMesh* out) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    int vd = in->numVerts;
    int fd = in->numFaces;
    int ed = in->numEdges;
    int hd = in->numHalfEdges;
    for(int i = h; i < hd; i += stride) {
        quadRefineEdges(i, in, out, vd, fd, ed);
    }    
}

__global__ void quadFacePoints(DeviceMesh* in, DeviceMesh* out) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int hd = in->numHalfEdges;
    for(int i = h; i < hd; i += stride) {
        quadFacePoint(i, in, out);
    } 
}

__global__ void quadEdgePoints(DeviceMesh* in, DeviceMesh* out) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int hd = in->numHalfEdges;
    for(int i = h; i < hd; i += stride) {
        quadEdgePoint(i, in, out);
    } 
}

__global__ void quadVertexPoints(DeviceMesh* in, DeviceMesh* out, int v0) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int hd = in->numHalfEdges;
    for(int i = h; i < hd; i += stride) {
        float n = valenceQuad(i, in);
        if(n > 0) {
            quadVertexPoint(i, in, out, n);
        } else if(in->twins[i] < 0) {
            quadBoundaryVertexPoint(i, in, out);
        }
    }   
}
