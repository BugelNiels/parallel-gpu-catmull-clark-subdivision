#include "kernelUtils.cuh"
#include "stdio.h"

__device__ int valence(int h, DeviceMesh* in) {
  int ht = in->twins[h];
  if (ht < 0) {
    return -1;
  }
  int n = 1;
  int hp = in->nexts[ht];
  while (hp != h) {
    if (hp < 0) {
      return -1;
    }
    ht = in->twins[hp];
    if (ht < 0) {
      return -1;
    }
    hp = in->nexts[ht];
    n++;
  }
  return n;
}

__device__ int valenceQuad(int h, DeviceMesh* in) {
  int ht = in->twins[h];
  if (ht < 0) {
    return -1;
  }
  int n = 1;
  int hp = next(ht);
  while (hp != h) {
    if (hp < 0) {
      return -1;
    }
    ht = in->twins[hp];
    if (ht < 0) {
      return -1;
    }
    hp = next(ht);
    n++;
  }
  return n;
}

__device__ int cycleLength(int h, DeviceMesh* in) {
  int m = 1;
  int hp = in->nexts[h];
  while(hp != h) {
    hp = in->nexts[hp];
    m++;
  }
  return m;
}