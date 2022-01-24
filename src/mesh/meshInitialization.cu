#include "../util/list.cuh"
#include "../util/util.cuh"
#include "meshInitialization.cuh"

/**
 * @brief Generates a unique integer from the provided two integers. Cantor pairing function.
 *
 * @param a First integer
 * @param b First integer
 * @return int Unique integer constructed from a and b
 */
int pairingFunction(int a, int b) { return 0.5 * (a + b) * (a + b + 1) + b; }

/**
 * @brief Creates a unique id for the undirected edge a-b. Regardless of the order of a and b, it will always generate
 * the same unique number.
 *
 * @param a Index of vertex a
 * @param b Index of vertex b
 * @return int A unique id for this edge
 */
int createUndirectedEdgeId(int a, int b) {
    if (a > b) {
        return pairingFunction(b, a);
    }
    return pairingFunction(a, b);
}

/**
 * @brief Ses the edge and twins correctly for the provided half-edge.
 *
 * @param mesh The half-edge mesh
 * @param h Index of the half-edge to set the edge and twins of
 * @param vertIdx2 VERT(NEXT(h))
 * @param edgeList List of all edges in the mesh so far
 */
void setEdgeAndTwins(Mesh* mesh, int h, int vertIdx2, List* edgeList) {
    int vertIdx1 = mesh->verts[h];
    int currentEdge = createUndirectedEdgeId(vertIdx1, vertIdx2);

    int edgeIdx = indexOf(edgeList, currentEdge);
    // edge does not exist yet
    if (edgeIdx == -1) {
        // same as doing this after appending and adding -1
        mesh->edges[h] = listSize(edgeList);
        append(edgeList, currentEdge);
    } else {
        mesh->edges[h] = edgeIdx;
        // edge already existed, meaning there is a twin somewhere earlier in the
        // list of edges
        int twin = indexOfArr(mesh->edges, mesh->numHalfEdges, edgeIdx);
        mesh->twins[h] = twin;
        mesh->twins[twin] = h;
    }
}

/**
 * @brief Adds a half-edge and its properties to the mesh
 *
 * @param mesh The half-edge mesh
 * @param h The half-edge index to add
 * @param faceIdx The face index the half-edge it belongs to
 * @param faceIndices Array of face indices
 * @param i Which half-edge in the face this is about. Can be at most valency-1
 * @param valency The valency of the face
 * @param edgeList The list of all edges in the mesh so far
 * @param isQuad 1 if the mesh is a quad, 0 otherwise
 */
void addHalfEdge(Mesh* mesh, int h, int faceIdx, int* faceIndices, int i, int valency, List* edgeList, int isQuad) {
    // vert
    int vertIdx = faceIndices[i];
    mesh->verts[h] = vertIdx;

    // twin
    mesh->twins[h] = -1;
    int nextVertIdx = faceIndices[(i + 1) % valency];
    setEdgeAndTwins(mesh, h, nextVertIdx, edgeList);

    if (isQuad == 0) {
        // prev and next
        int prev = h - 1;
        int next = h + 1;
        if (i == 0) {
            // prev = h - 1 + faceValency
            prev += valency;
        } else if (i == valency - 1) {
            // next = h + 1 - faceValency
            next -= valency;
        }
        mesh->prevs[h] = prev;
        mesh->nexts[h] = next;
        // face
        mesh->faces[h] = faceIdx;
    }
}

/**
 * @brief Allocates memory for the mesh and copies data from the obj file into it
 *
 * @param mesh The mesh in which to allocate memory
 */
void allocateMeshMemory(Mesh* mesh) {
    mesh->xCoords = (float*)malloc(mesh->numVerts * sizeof(float));
    mesh->yCoords = (float*)malloc(mesh->numVerts * sizeof(float));
    mesh->zCoords = (float*)malloc(mesh->numVerts * sizeof(float));

    mesh->twins = (int*)malloc(mesh->numHalfEdges * sizeof(int));
    mesh->verts = (int*)malloc(mesh->numHalfEdges * sizeof(int));
    mesh->edges = (int*)malloc(mesh->numHalfEdges * sizeof(int));

    if (mesh->isQuad == 0) {
        mesh->nexts = (int*)malloc(mesh->numHalfEdges * sizeof(int));
        mesh->prevs = (int*)malloc(mesh->numHalfEdges * sizeof(int));
        mesh->faces = (int*)malloc(mesh->numHalfEdges * sizeof(int));
    } else {
        mesh->nexts = NULL;
        mesh->prevs = NULL;
        mesh->faces = NULL;
    }
}

/**
 * @brief Creates a mesh from the provided objfile
 *
 * @param obj The objfile file to convert into a mesh struct
 * @return Mesh The mesh
 */
Mesh meshFromObjFile(ObjFile obj) {
    Mesh mesh;
    // Set initial properties
    mesh.numFaces = obj.numFaces;
    mesh.numVerts = obj.numVerts;
    mesh.isQuad = obj.isQuad;

    mesh.numHalfEdges = 0;
    for (int faceIdx = 0; faceIdx < obj.numFaces; ++faceIdx) {
        mesh.numHalfEdges += obj.faceValencies[faceIdx];
    }

    allocateMeshMemory(&mesh);
    memcpy(mesh.xCoords, obj.xCoords, mesh.numVerts * sizeof(float));
    memcpy(mesh.yCoords, obj.yCoords, mesh.numVerts * sizeof(float));
    memcpy(mesh.zCoords, obj.zCoords, mesh.numVerts * sizeof(float));

    int h = 0;
    List edgeList = initEmptyList();
    // loop over every face
    for (int faceIdx = 0; faceIdx < obj.numFaces; ++faceIdx) {
        int* faceIndices = obj.faceIndices[faceIdx];
        // each face will end up with a number of half edges equal to its number of
        // faces
        int valency = obj.faceValencies[faceIdx];
        for (int i = 0; i < valency; ++i) {
            addHalfEdge(&mesh, h, faceIdx, faceIndices, i, valency, &edgeList, obj.isQuad);
            h++;
        }
    }
    mesh.numEdges = listSize(&edgeList);
    printf("Created Mesh with: %d half-edges, %d faces, %d vertices and %d edges\n\n", mesh.numHalfEdges, mesh.numFaces,
           mesh.numVerts, mesh.numEdges);
    return mesh;
}