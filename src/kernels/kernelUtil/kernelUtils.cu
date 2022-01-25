#include "kernelUtils.cuh"
#include "stdio.h"

/**
 * @brief Calculates the valence of the vertex originating from the provided half-edge
 *
 * @param h The index of the half-edge
 * @param in Mesh in which the vertex exists
 * @return Valence of vert(h)
 */
__device__ int valence(int h, DeviceMesh* in) {
    int ht = in->twins[h];
    if (ht < 0) {
        return -1;
    }
    int n = 1;
    int hp = in->nexts[ht];
    while (hp != h) {
        ht = in->twins[hp];
        if (ht < 0) {
            return -1;
        }
        hp = in->nexts[ht];
        n++;
    }
    return n;
}

/**
 * @brief Calculates the valence of the vertex originating from the provided half-edge
 *
 * @param h The index of the half-edge
 * @param in Quad mesh in which the vertex exists
 * @return Valence of vert(h)
 */
__device__ int valenceQuad(int h, DeviceMesh* in) {
    int ht = in->twins[h];
    if (ht < 0) {
        return -1;
    }
    int n = 1;
    int hp = next(ht);
    while (hp != h) {
        ht = in->twins[hp];
        if (ht < 0) {
            return -1;
        }
        hp = next(ht);
        n++;
    }
    return n;
}

/**
 * @brief Calculates the cycle length of a face. Equivalent to the number of half-edges of a face
 *
 * @param h The index of the half-edge
 * @param in Mesh in which the face exists
 * @return Cycle length of face(h)
 */
__device__ int cycleLength(int h, DeviceMesh* in) {
    int m = 1;
    int hp = in->nexts[h];
    while (hp != h) {
        hp = in->nexts[hp];
        m++;
    }
    return m;
}