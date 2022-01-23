#include "../util/list.cuh"
#include "../util/util.cuh"
#include "meshInitialization.cuh"

int pairingFunction(int a, int b) {
    // cantor pairing function
    return 0.5 * (a + b) * (a + b + 1) + b;
}

int createUndirectedEdgeId(int a, int b) {
    if (a > b) {
        return pairingFunction(b, a);
    }
    return pairingFunction(a, b);
}

void setEdgeAndTwins(Mesh* mesh, int h, int vertIdx2, List* edgeList, int n) {
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
        int twin = indexOfArr(mesh->edges, n, edgeIdx);
        mesh->twins[h] = twin;
        mesh->twins[twin] = h;
    }
}

void addHalfEdge(Mesh* mesh, int h, int faceIdx, int* faceIndices, int i, int valency, List* edgeList, int isQuad,
                 int n) {
    // vert
    int vertIdx = faceIndices[i];
    mesh->verts[h] = vertIdx;

    // twin
    mesh->twins[h] = -1;
    int nextVertIdx = faceIndices[(i + 1) % valency];
    setEdgeAndTwins(mesh, h, nextVertIdx, edgeList, n);

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

Mesh meshFromObjFile(ObjFile obj) {
    Mesh mesh;
    mesh.numFaces = obj.numFaces;
    mesh.numVerts = obj.numVerts;

    int numHalfEdges = 0;
    for (int faceIdx = 0; faceIdx < obj.numFaces; ++faceIdx) {
        numHalfEdges += obj.faceValencies[faceIdx];
    }
    mesh.numHalfEdges = numHalfEdges;

    mesh.xCoords = (float*)malloc(mesh.numVerts * sizeof(float));
    mesh.yCoords = (float*)malloc(mesh.numVerts * sizeof(float));
    mesh.zCoords = (float*)malloc(mesh.numVerts * sizeof(float));
    memcpy(mesh.xCoords, obj.xCoords, mesh.numVerts * sizeof(float));
    memcpy(mesh.yCoords, obj.yCoords, mesh.numVerts * sizeof(float));
    memcpy(mesh.zCoords, obj.zCoords, mesh.numVerts * sizeof(float));

    mesh.twins = (int*)malloc(numHalfEdges * sizeof(int));
    mesh.verts = (int*)malloc(numHalfEdges * sizeof(int));
    mesh.edges = (int*)malloc(numHalfEdges * sizeof(int));

    mesh.isQuad = obj.isQuad;
    if (obj.isQuad == 0) {
        mesh.nexts = (int*)malloc(numHalfEdges * sizeof(int));
        mesh.prevs = (int*)malloc(numHalfEdges * sizeof(int));
        mesh.faces = (int*)malloc(numHalfEdges * sizeof(int));
    } else {
        mesh.nexts = NULL;
        mesh.prevs = NULL;
        mesh.faces = NULL;
    }
    int h = 0;
    List edgeList = initEmptyList();
    // loop over every face
    for (int faceIdx = 0; faceIdx < obj.numFaces; ++faceIdx) {
        int* faceIndices = obj.faceIndices[faceIdx];
        // each face will end up with a number of half edges equal to its number of
        // faces
        int valency = obj.faceValencies[faceIdx];
        for (int i = 0; i < valency; ++i) {
            addHalfEdge(&mesh, h, faceIdx, faceIndices, i, valency, &edgeList, obj.isQuad, numHalfEdges);
            h++;
        }
    }
    mesh.numEdges = listSize(&edgeList);
    printf("Created Mesh with: %d half-edges, %d faces, %d vertices and %d edges\n\n", mesh.numHalfEdges, mesh.numFaces,
           mesh.numVerts, mesh.numEdges);
    return mesh;
}