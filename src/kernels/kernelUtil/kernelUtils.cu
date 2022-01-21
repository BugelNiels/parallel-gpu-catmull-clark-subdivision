#include "kernelUtils.cuh"
#include "stdio.h"

__device__ int forwardValence(int h, DeviceMesh* in) {
  int ht = in->twins[h];
  if (ht < 0) {
    return 0;
  }
  int n = 1;
  int hp = in->nexts[ht];
  while (hp != h) {
    if (hp < 0) {
      return n;
    }
    ht = in->twins[hp];
    if (ht < 0) {
      return n;
    }
    hp = in->nexts[ht];
    n++;
  }
  return n;
}

__device__ int backValence(int h, DeviceMesh* in) {
  int hp = in->twins[in->prevs[h]];
  int n = 0;
  while (hp != h) {    
    if (hp < 0) {
      return n;
    }
    hp = in->twins[in->prevs[hp]];
    n++;
  }
  return n;
}

__device__ int boundaryValence(int h, DeviceMesh* in) {
  return 1 + forwardValence(h, in) + backValence(h, in);
}

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


__device__ int forwardValenceQuad(int h, DeviceMesh* in) {
  int ht = in->twins[h];
  if (ht < 0) {
    return 0;
  }
  int n = 1;
  int hp = next(ht);
  while (hp != h) {
    if (hp < 0) {
      return n;
    }
    ht = in->twins[hp];
    if (ht < 0) {
      return n;
    }
    hp = next(ht);
    n++;
  }
  return n;
}

__device__ int backValenceQuad(int h, DeviceMesh* in) {
  int hp = in->twins[prev(h)];
  int n = 1;
  while (hp != h) {    
    if (hp < 0) {
      return n;
    }
    hp = in->twins[prev(hp)];
    n++;
  }
  return n;
}

__device__ int boundaryValenceQuad(int h, DeviceMesh* in) {
  return 1 + forwardValenceQuad(h, in) + backValenceQuad(h, in);
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