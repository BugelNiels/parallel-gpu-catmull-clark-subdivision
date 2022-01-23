#include <stdio.h>
#include <stdlib.h>

#include "mesh.cuh"

Mesh makeEmptyCopy(Mesh* mesh) { return initMesh(mesh->numVerts, mesh->numHalfEdges, mesh->numFaces, mesh->numEdges); }

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

void allocQuadMesh(Mesh* mesh) {
    mesh->xCoords = (float*)malloc(mesh->numVerts * sizeof(float));
    mesh->yCoords = (float*)malloc(mesh->numVerts * sizeof(float));
    mesh->zCoords = (float*)malloc(mesh->numVerts * sizeof(float));
    mesh->twins = (int*)malloc(mesh->numHalfEdges * sizeof(int));
    mesh->edges = (int*)malloc(mesh->numHalfEdges * sizeof(int));
    mesh->verts = (int*)malloc(mesh->numHalfEdges * sizeof(int));
}

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
