#ifndef _STATS_H_
#define _STATS_H_

typedef struct {
   float x,y,z;
} XYZ;

typedef struct {
   XYZ p[3];         /* Vertices */
   // XYZ c;            /* Centroid */
   // XYZ n[3];         /* Normal   */
} TRIANGLE;

// compute the volume of the current object body
void computeVolume(TRIANGLE* vertices, int nTriangles, int* index, float* mass, XYZ* cm);

#endif