#ifndef DEVICE_MESH_CUH
#define DEVICE_MESH_CUH

#include "mesh.cuh"

/**
 * @brief Mesh structure for usage on the GPU device
 * 
 */
typedef struct DeviceMesh {
    // these have length numVerts
    float* xCoords;
    float* yCoords;
    float* zCoords;

    // these all have length numHalfEdges
    int* twins;
    int* nexts;
    int* prevs;
    int* verts;
    int* edges;
    int* faces;

    // general mesh properties
    int numHalfEdges;
    int numEdges;
    int numFaces;
    int numVerts;
} DeviceMesh;

DeviceMesh createEmptyCopyOnDevice(Mesh* mesh);
DeviceMesh initEmptyDeviceMesh(int numVerts, int numHalfEdges, int numFaces, int numEdges);

DeviceMesh* toDevicePointer(DeviceMesh* mesh_h);
DeviceMesh devicePointerToHostMesh(DeviceMesh* mesh_d);
void freeDeviceMesh(DeviceMesh* mesh);

#endif  // DEVICE_MESH_CUH