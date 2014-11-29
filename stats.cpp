#include "stats.h"
#include <stdio.h>
#include <stdlib.h>

#define Subexpressions(w0,w1,w2,f1,f2,f3,g0,g1,g2) {float temp0 = w0 + w1; f1 = temp0 + w2; float temp1 = w0 * w0;float temp2 = temp1 + w1 * temp0;f2 = temp2 + w2 * f1; f3 = w0 * temp1 + w1 * temp2 + w2 * f2; g0 = f2 + w0 * (f1 + w0); g1 = f2 + w1 * (f1 + w1); g2 = f2 + w2 * (f1 + w2);}

// compute the volume of the current object body
void computeVolume(TRIANGLE* vertices, int nTriangles, int* index, float* mass, XYZ* cm)
{
	const float mult[10] = {1.0/6.0, 1.0/24.0, 1.0/24.0, 1.0/24.0, 1.0/60.0, 1.0/60.0, 1.0/60.0, 1.0/120.0, 1.0/120.0, 1.0/120.0};

	float intg[10] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

	int t = 0;

	for (t = 0; t < nTriangles; t++)
	{
		// get vertices of triangle t
		int i0;//, i1, i2;
		i0 = index[3*t];// i1 = index[3*t+1]; i2 = index[3*t+2];

		float x0, x1, x2;
		float y0, y1, y2;
		float z0, z1, z2;

		x0 = vertices[i0].p[0].x; y0 = vertices[i0].p[0].y; z0 = vertices[i0].p[0].z;
		x1 = vertices[i0].p[1].x; y1 = vertices[i0].p[1].y; z1 = vertices[i0].p[1].z;
		x2 = vertices[i0].p[2].x; y2 = vertices[i0].p[2].y; z2 = vertices[i0].p[2].z;

		// get edges and cross product of edges
		float a1, b1, c1, a2, b2, c2, d0, d1, d2;
		a1 = x1 - x0; b1 = y1 - y0; c1 = z1 - z0;
		a2 = x2 - x0; b2 = y2 - y0; c2 = z2 - z0;
		d0 = b1 * c2 - b2 * c1;
		d1 = a2 * c1 - a1 * c2;
		d2 = a1 * b2 - a2 * b1;

		// compute integral terms
		float f1x,f2x,f3x,g0x,g1x,g2x;
		float f1y,f2y,f3y,g0y,g1y,g2y;
		float f1z,f2z,f3z,g0z,g1z,g2z;

		// float temp0 = x0 + x1; 
		// f1x = temp0 + x2;
		Subexpressions(x0,x1,x2,f1x,f2x,f3x,g0x,g1x,g2x);
		Subexpressions(y0,y1,y2,f1y,f2y,f3y,g0y,g1y,g2y);
		Subexpressions(z0,z1,z2,f1z,f2z,f3z,g0z,g1z,g2z);
		
		intg[0] += d0*f1x;
		intg[1] += d0*f2x; 
		intg[2] += d1*f2y; 
		intg[3] += d2*f2z;
		intg[4] += d0*f3x;
		intg[5] += d1*f3y;
		intg[6] += d2*f3z;
		intg[7] += d0*(y0*g0x+y1*g1x+y2*g2x);
		intg[8] += d1*(z0*g0y+z1*g1y+z2*g2y);
		intg[9] += d2*(x0*g0z+x1*g1z+x2*g2z);
	}

	int i;
	for (i = 0; i < 10; i++)
	{
		intg[i] *= mult[i];
	}

	*mass = intg[0];

	// center of mass
	if (*mass != 0.0)
	{
		cm->x = intg[1] / *mass;
		cm->y = intg[2] / *mass;
		cm->z = intg[3] / *mass;	
	}
}