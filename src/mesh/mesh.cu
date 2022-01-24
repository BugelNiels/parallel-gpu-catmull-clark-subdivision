#include <stdio.h>
#include <stdlib.h>

#include "mesh.cuh"

/**
 * @brief Creates an empty copy of the provided mesh.
 * Only copies the number of vertices, half-edges, faces and edges.
 *
 * @param mesh The mesh whose properties to copy
 * @return Mesh An empty mesh
 */
Mesh makeEmptyCopy(Mesh* mesh) { return initMesh(mesh->numVerts, mesh->numHalfEdges, mesh->numFaces, mesh->numEdges); }

/**
 * @brief Initializes a mostly empty mesh with no memory allocated
 *
 * @param numVerts Number of vertices
 * @param numHalfEdges Number of half-edges
 * @param numFaces Number of faces
 * @param numEdges Number of edges
 * @return Mesh An empty mesh
 */
Mesh initMesh(int numVerts, int numHalfEdges, int numFaces, int numEdges) {
    Mesh mesh;
    mesh.numVerts = numVerts;
    mesh.numHalfEdges = numHalfEdges;
    mesh.numEdges = numEdges;
    mesh.numFaces = numFaces;
    mesh.xCoords = NULL;
    mesh.yCoords = NULL;
    mesh.zCoords = NULL;
    mesh.nexts = NULL;
    mesh.prevs = NULL;
    mesh.faces = NULL;
    mesh.twins = NULL;
    mesh.edges = NULL;
    mesh.verts = NULL;
    return mesh;
}

/**
 * @brief Allocates enough memory for a quad mesh. The provided quad mesh should have the numVerts and numHalfEdges
 * properties set appropriately.
 *
 * @param mesh The quad mesh for which to allocate memory.
 */
void allocQuadMesh(Mesh* mesh) {
    mesh->xCoords = (float*)malloc(mesh->numVerts * sizeof(float));
    mesh->yCoords = (float*)malloc(mesh->numVerts * sizeof(float));
    mesh->zCoords = (float*)malloc(mesh->numVerts * sizeof(float));
    mesh->twins = (int*)malloc(mesh->numHalfEdges * sizeof(int));
    mesh->edges = (int*)malloc(mesh->numHalfEdges * sizeof(int));
    mesh->verts = (int*)malloc(mesh->numHalfEdges * sizeof(int));
}

/**
 * @brief Frees the memory taken by the provided mesh
 *
 * @param mesh The mesh whose data to free
 */
void freeMesh(Mesh* mesh) {
    free(mesh->xCoords);
    free(mesh->yCoords);
    free(mesh->zCoords);
    free(mesh->twins);
    free(mesh->nexts);
    free(mesh->prevs);
    free(mesh->verts);
    free(mesh->edges);
    free(mesh->faces);
}

/**
 * @brief Writes data from a mesh to the object file. Assumes a quad mesh.
 *
 * @param mesh The mesh to write to an object file
 * @param path Path of where to write the mesh to
 */
void toObjFile(Mesh* mesh, char* path) {
    printf("Writing mesh to file..\n");
    FILE* objFile = fopen(path, "w");
    if (objFile == NULL) {
        printf("Error opening or creating .obj file!\n");
        exit(1);
    }
    // print vertices
    for (int v = 0; v < mesh->numVerts; v++) {
        fprintf(objFile, "v %.6lf %.6lf %.6lf\n", mesh->xCoords[v], mesh->yCoords[v], mesh->zCoords[v]);
    }
    fprintf(objFile, "# Numfaces: %d\n\n", mesh->numFaces);
    // list of face indices
    for (int f = 0; f < mesh->numFaces; f++) {
        fprintf(objFile, "f");
        for (int v = 0; v < 4; v++) {
            // indices in .obj start at 1
            fprintf(objFile, " %d", mesh->verts[f * 4 + v] + 1);
        }
        fprintf(objFile, "\n");
    }
    fclose(objFile);
}
