#ifndef MESH_CUH
#define MESH_CUH

/**
 * @brief Mesh structure for usage on the host device
 * 
 */
typedef struct Mesh {
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

    int numHalfEdges;
    int numEdges;
    int numFaces;
    int numVerts;
    // 1 if the mesh is a quad; 0 otherwise
    int isQuad;
} Mesh;

Mesh makeEmptyCopy(Mesh* mesh);
Mesh initMesh(int numVerts, int numHalfEdges, int numFaces, int numEdges);
void freeMesh(Mesh* mesh);
void toObjFile(Mesh* mesh, char* path);
void allocQuadMesh(Mesh* mesh);

#endif  // MESH_CUH