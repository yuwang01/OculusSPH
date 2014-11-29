#define ABS(x) (x < 0 ? -(x) : (x))

kernel void sph_kernel_applyBodyForce (
		int kParticleCount,
		__global float4* vel,
		float dt)
{
	size_t gid = get_global_id(0);

	vel[gid].y -= 9.8 * dt;

    barrier(CLK_LOCAL_MEM_FENCE);
}

kernel void sph_kernel_advance(
		int kParticleCount,
		float dt,
		__global float4* position,
		__global float4* prePos,
		__global float4* velocity)
{
	size_t gid = get_global_id(0);

	prePos[gid].xyzw = position[gid].xyzw;

    barrier(CLK_LOCAL_MEM_FENCE);

	position[gid].xyz += (float3)(dt) * velocity[gid].xyz;

}

kernel void sph_kernel_hashparticles(
		float viewWidth,
		float viewHeight,
		float viewDepth,
		int nx,
		int ny,
		int nz,
		float cellSize,
		__global float4* pos,
		__global uint2* partIdx,
		__global float4* vel)
{
	size_t gid = get_global_id(0);

	float3 tempPos;
	// pos(x, y, z) in [-viewWidth/2, viewWidth/2] mapped to [0, nx*cellSize]
    // newPos = (val - src0) / (src1 - src0) * (dst1 - dst0) + dst0
    float rangeRatioX = (nx * cellSize - 0.0f) / viewWidth;
    float rangeRatioY = (ny * cellSize - 0.0f) / viewHeight;
    float rangeRatioZ = (nz * cellSize - 0.0f) / viewDepth;
        
    tempPos.x = (pos[gid].x - (-viewWidth/2) ) * rangeRatioX;
    tempPos.y = (pos[gid].y - (-viewHeight/2)) * rangeRatioY;
    tempPos.z = (pos[gid].z - (-viewDepth/2) ) * rangeRatioZ;

    // myPos(x, y, z) in [0, nx*cellSize] mapped to 3d voxel coords [v3d.x, v3d.y, v3d.z]
    // each between the range [0, n] (assuming cubic space)
    float3 voxel3d;
    voxel3d.x = tempPos.x / cellSize;
    voxel3d.y = tempPos.y / cellSize;
    voxel3d.z = tempPos.z / cellSize;

    int voxelID = floor(voxel3d.x) + floor(voxel3d.y) * nx + floor(voxel3d.z) * nx * ny;
        
    pos[gid].w = voxelID;
        
    partIdx[gid].x = voxelID;
    partIdx[gid].y = gid;	

}

kernel void sph_kernel_sort(
		__global uint2* partIdx,
		int stage,
		int passOfStage,
		int direction)
{
	uint sortIncreasing = direction;
	size_t threadId = get_global_id(0);
    
    uint pairDistance = 1 << (stage - passOfStage);
    uint blockWidth   = 2 * pairDistance;

    uint leftId = (threadId % pairDistance) 
    				+ (threadId / pairDistance) * blockWidth;

	uint rightId = leftId + pairDistance;
    
	uint2 leftElement = partIdx[leftId];
    uint2 rightElement = partIdx[rightId];
    
    uint sameDirectionBlockWidth = 1 << stage;
    
	if((threadId/sameDirectionBlockWidth) % 2 == 1)
    	sortIncreasing = 1 - sortIncreasing;

	uint2 greater;
	uint2 lesser;
    if(leftElement.x > rightElement.x)
    {
    	greater = leftElement;
        lesser  = rightElement;
	}
    else
    {
    	greater = rightElement;
        lesser  = leftElement;
	}
    
    if(sortIncreasing)
    {
    	partIdx[leftId]  = lesser;
        partIdx[rightId] = greater;
	}
    else
    {
    	partIdx[leftId]  = greater;
        partIdx[rightId] = lesser;
	}

}

kernel void sph_kernel_sortPostPass(
		__global float4* pos,
        __global float4* vel,
        __global uint2* partIdx,
        __global float4* sortedPos,
        __global float4* sortedVel,
        __global float4* prevPos,
        __global float4* sortedPrevPos)
{
	   unsigned int gid = get_global_id(0);

        int particleID = partIdx[gid].y;

        // float4 tempPos = pos[particleID];

        sortedPos[gid].x = pos[particleID].x;
        sortedPos[gid].y = pos[particleID].y;
        sortedPos[gid].z = pos[particleID].z;
        sortedPos[gid].w = pos[particleID].w;

        // float4 tempVel = vel[particleID];

        sortedVel[gid].x = vel[particleID].x;
        sortedVel[gid].y = vel[particleID].y;
        sortedVel[gid].z = vel[particleID].z;
        sortedVel[gid].w = vel[particleID].w;

        // float4 tempPrevPos = prevPos[particleID];

        sortedPrevPos[gid].x = prevPos[particleID].x;
        sortedPrevPos[gid].y = prevPos[particleID].y;
        sortedPrevPos[gid].z = prevPos[particleID].z;
        sortedPrevPos[gid].w = prevPos[particleID].w;

        barrier(CLK_GLOBAL_MEM_FENCE);

        pos[gid].x = sortedPos[gid].x;
        pos[gid].y = sortedPos[gid].y;
        pos[gid].z = sortedPos[gid].z;
        pos[gid].w = sortedPos[gid].w;

        vel[gid].x = sortedVel[gid].x;
        vel[gid].y = sortedVel[gid].y;
        vel[gid].z = sortedVel[gid].z;
        vel[gid].w = sortedVel[gid].w;
        
        pos[particleID].x = sortedPos[particleID].x;
        pos[particleID].y = sortedPos[particleID].y;
        pos[particleID].z = sortedPos[particleID].z;
        pos[particleID].w = sortedPos[particleID].w;

        barrier(CLK_GLOBAL_MEM_FENCE);

        vel[particleID].x = sortedVel[particleID].x;
        vel[particleID].y = sortedVel[particleID].y;
        vel[particleID].z = sortedVel[particleID].z;
        vel[particleID].w = sortedVel[particleID].w;
}

kernel void kernel_copy(
        __global float4* target,
        __global float4* source
    )
{
    size_t gid = get_global_id(0);

    target[gid].x = source[gid].x;
    target[gid].y = source[gid].y;
    target[gid].z = source[gid].z;
    target[gid].w = source[gid].w;
}

kernel void sph_kernel_indexx(
		int kParticleCount,
		__global float4* pos,
		__global int* gridCellIdx)
{
	size_t gid = get_global_id(0);
	gridCellIdx[gid] = -1;

	int low = 0;
	int hi = kParticleCount - 1;
    int mid = 0;

    while (low <= hi) {
    	mid = (hi + low) / 2;
        // if (floor(sortedPos[mid].w) == gid) {
        if (floor(pos[mid].w) == gid) {    
        	int front = mid - 1;
                
            // while((front >= 0) && (floor(sortedPos[front].w) == gid))
            while((front >= 0) && (floor(pos[front].w) == gid))
            {
            	front--;
			}

			if (mid > front - 1)
            {
            	gridCellIdx[gid] = front + 1;
			}

            break;

            // } else if ( sortedPos[mid].w < gid) {
			} else if ( pos[mid].w < gid) {                
                low = mid + 1;
            } else {
                hi = mid - 1;
            }
	}
}

kernel void sph_kernel_indexPostPass(
		int kParticleCount,
        __global int* gridCellIdx,
        __global int* gridCellIdxFixedUp)
{
	size_t gid = get_global_id(0);

	if (gridCellIdx[gid] != -1)
    {
    	gridCellIdxFixedUp[gid] = gridCellIdx[gid];
	} else {
    	int preCell = gid;
        
        while (preCell >= 0) 
        {
        	int pid = gridCellIdx[preCell];
            if (pid != -1)
            {
            	gridCellIdxFixedUp[gid] = pid;
                break;
			}
            else {
            	preCell--;
			}
		}
	}	
}

kernel void sph_kernel_findNeighbors(
		int kParticleCount,
        float viewW,
        float viewH,
        float viewD,
        int nx,
        int ny,
        int nz,
        float cellSize,
        __global float4* pos,
        // __global int* gridCellIdxFixedUp,
        __global int* neighborMap,
        int maxNeighbors,
        float kEpsilon)
{
	size_t gid = get_global_id(0);

	float3 myPos = pos[gid].xyz;

	int count = 0;
    int i;
    int j;

    // set all entries to be the particle itself
    for (i = 0; i < maxNeighbors; i++)
    {
    	neighborMap[maxNeighbors*gid+i] = gid;
	}

    for (j = 0; j < kParticleCount; j++)
    {
    	// float3 particleJ = sortedPos[j].xyz;
        float3 particleJ = pos[j].xyz;
        float dist = length(myPos.xyz - particleJ);

        if ((dist >= kEpsilon) && (dist <= cellSize) && (j>=0) && ((unsigned int)j != gid))
        {
        	neighborMap[maxNeighbors*gid+count] = j;
            count++;
            if (count == maxNeighbors)
            	break;
		}
	}
}

kernel void sph_kernel_pressure(
		float m,
        float cellSize,
        float kNorm,
        float kNearNorm,
        int maxNeighbors,
        float kStiffness,
        float kNearStiffness,
        float kRestDensity,
        float kEpsilon,
        __global float4* pos,
        __global int* neighborMap,
        __global float* density,
        __global float* nearDensity,
        __global float* pressure,
        __global float* nearPressure)
{
	size_t gid = get_global_id(0);

	float3 myPos = pos[gid].xyz;

    int i;
    float3 posJ;
    float tempDens = 0.0;
    float tempNearDens = 0.0;

    for (i = 0; i < maxNeighbors; i++)
    {
    	int nID = neighborMap[maxNeighbors * gid+i];
        if ((nID >= 0) && ((unsigned int)nID == gid))
        	continue;

		posJ = pos[nID].xyz;

        float r = length(myPos - posJ);

        float a = 1 - r/cellSize;

        tempDens += m * a * a * a * kNorm;
        tempNearDens += m * a * a * a * a * kNearNorm;

	}

    density[gid] = tempDens;
    nearDensity[gid] = tempNearDens;
    pressure[gid] = kStiffness * (tempDens - m * kRestDensity);
    nearPressure[gid] = kNearStiffness * tempNearDens;	
}

kernel void mc_kernel_reset(
    int numVolIdx,
    __global float4* mcgrid)
{
    size_t gid = get_global_id(0);

    mcgrid[gid].w = 0.0;
}

kernel void mc_kernel_gridval(
		float m,
        float cellSize,
        float kNorm,
        float kNearNorm,
        float kEpsilon,
        __global float4* pos,
        __global float4* mcgrid,
        int numVolIdx)
{
	size_t gid = get_global_id(0);

	float3 myPos = pos[gid].xyz;

    int i;
    float3 posJ;
    
    float a;
    
    for (i = 0; i < numVolIdx; i++)
    {
        posJ = mcgrid[i].xyz;
        a = 0.0;

        float r = length(myPos - posJ);

        if (r >= kEpsilon && r <= cellSize)
        {
            // a = 1.0 - r/cellSize;
            // a = r/cellSize;
            a = cellSize/r;
        }
        else
            continue;

        // mcgrid[i].w += m * (.5*cos(a)) * kNearNorm;
        mcgrid[i].w += m * a * kNearNorm;
        // mcgrid[i].w += m * a * kNorm;
    }
}

kernel void mc_kernel_gridGrad(
        int volnx,
        int volny,
        int volnz,
        float edgex,
        float edgey,
        float edgez,
        __global float4* mcgrid,
        __global float4* mcgridGrad)
{
    size_t gid = get_global_id(0);
    
    if (mcgridGrad[gid].w != -1.0)
    {
        mcgridGrad[gid].x = (mcgrid[gid-1].w - mcgrid[gid+1].w)/edgex;
        mcgridGrad[gid].y = (mcgrid[gid-(volny+1)].w - mcgrid[gid+(volny+1)].w)/edgey;
        mcgridGrad[gid].z = (mcgrid[gid-(volny+1)*(volnx+1)].w - mcgrid[gid+(volny+1)*(volnx+1)].w)/edgez;
    }
}

kernel void mc_kernel_cubeindex(
		float isoval, // isosurface value
        int volnx,
        int volny,
        int volnz,
		__global float4* mcgrid,
        __global int* pointindex,
		__global int* cubeindex)
{
    size_t gid = get_global_id(0);

    if ((int)gid < volnx * volny * volnz)
    {
        int idx0 = pointindex[8*gid+0];
        int idx1 = pointindex[8*gid+1];
        int idx2 = pointindex[8*gid+2];
        int idx3 = pointindex[8*gid+3];
        int idx4 = pointindex[8*gid+4];
        int idx5 = pointindex[8*gid+5];
        int idx6 = pointindex[8*gid+6];
        int idx7 = pointindex[8*gid+7];

        float val0 = mcgrid[idx4].w;
        float val1 = mcgrid[idx5].w;
        float val2 = mcgrid[idx1].w;
        float val3 = mcgrid[idx0].w;
        float val4 = mcgrid[idx7].w;
        float val5 = mcgrid[idx6].w;
        float val6 = mcgrid[idx2].w;
        float val7 = mcgrid[idx3].w;

        int cidx = 0;

        cidx =  (val0 < isoval); 
        cidx += (val1 < isoval)*2; 
        cidx += (val2 < isoval)*4; 
        cidx += (val3 < isoval)*8; 
        cidx += (val4 < isoval)*16; 
        cidx += (val5 < isoval)*32; 
        cidx += (val6 < isoval)*64; 
        cidx += (val7 < isoval)*128;

        cubeindex[gid] = cidx; 
    }
}

static float3 VertexInterp(float isolevel, float3 p1, float3 p2, float valp1, float valp2)
{
   float mu;
   float3 p;

   if (ABS(isolevel-valp1) < 0.00001)
      return(p1);
   if (ABS(isolevel-valp2) < 0.00001)
      return(p2);
   if (ABS(valp1-valp2) < 0.00001)
      return(p1);

   mu = (isolevel - valp1) / (valp2 - valp1);
   p.x = p1.x + mu * (p2.x - p1.x);
   p.y = p1.y + mu * (p2.y - p1.y);
   p.z = p1.z + mu * (p2.z - p1.z);

   return(p);
}

static float3 GradInterp(float isolevel, float3 g1, float3 g2, float valp1, float valp2)
{
    float mu;
    float3 g;

    if (ABS(isolevel-valp1) < 0.00001)
        return (g1);
    if (ABS(isolevel-valp2) < 0.00001)
        return (g2);
    if (ABS(valp1-valp2) < 0.00001)
        return (g1);

    mu = (isolevel - valp1) / (valp2 - valp1);
    g.x = g1.x + mu * (g2.x - g1.x);
    g.y = g1.y + mu * (g2.y - g1.y);
    g.z = g1.z + mu * (g2.z - g1.z);

    return(g);
}

kernel void mc_kernel_interpolate(
        float isoval,
        int volnx,
        int volny,
        int volnz,
        __global int* pointindex,
        __global float4* mcgrid,
        __global float4* mcgridGrad,
        __global int* cubeIndex,
        __global int* edgeTable,
        __global int* triTable,
        __global float3* tri,
        __global int* triCount,
        __global float3* norm)
{
    size_t gid = get_global_id(0);

    if ((int)gid < volnx * volny * volnz)
    {
        int idx0 = pointindex[8*gid+0];
        int idx1 = pointindex[8*gid+1];
        int idx2 = pointindex[8*gid+2];
        int idx3 = pointindex[8*gid+3];
        int idx4 = pointindex[8*gid+4];
        int idx5 = pointindex[8*gid+5];
        int idx6 = pointindex[8*gid+6];
        int idx7 = pointindex[8*gid+7];

        float val0 = mcgrid[idx4].w;
        float val1 = mcgrid[idx5].w;
        float val2 = mcgrid[idx1].w;
        float val3 = mcgrid[idx0].w;
        float val4 = mcgrid[idx7].w;
        float val5 = mcgrid[idx6].w;
        float val6 = mcgrid[idx2].w;
        float val7 = mcgrid[idx3].w;

        float3 P0, P1, P2, P3, P4, P5, P6, P7;
        P0 = mcgrid[idx4].xyz;
        P1 = mcgrid[idx5].xyz;
        P2 = mcgrid[idx1].xyz;
        P3 = mcgrid[idx0].xyz;
        P4 = mcgrid[idx7].xyz;
        P5 = mcgrid[idx6].xyz;
        P6 = mcgrid[idx2].xyz;
        P7 = mcgrid[idx3].xyz;

        float3 G0, G1, G2, G3, G4, G5, G6, G7;
        G0 = mcgridGrad[idx4].xyz;
        G1 = mcgridGrad[idx5].xyz;
        G2 = mcgridGrad[idx1].xyz;
        G3 = mcgridGrad[idx0].xyz;
        G4 = mcgridGrad[idx7].xyz;
        G5 = mcgridGrad[idx6].xyz;
        G6 = mcgridGrad[idx2].xyz;
        G7 = mcgridGrad[idx3].xyz;

        __local float3 vertlist[12];
        __local float3 normlist[12];
        
        int edgetable = edgeTable[cubeIndex[gid]];
        if (edgetable != 0)
        {
            if (edgetable & 1)
            {
                vertlist[0] = VertexInterp(isoval, P0, P1, val0, val1);
                normlist[0] = GradInterp(isoval, G0, G1, val0, val1);
            }

            if (edgetable & 2)
            {
                vertlist[1] = VertexInterp(isoval, P1, P2, val1, val2);
                normlist[1] = GradInterp(isoval, G1, G2, val1, val2);
            }

            if (edgetable & 4)
            {
                vertlist[2] = VertexInterp(isoval, P2, P3, val2, val3);
                normlist[2] = GradInterp(isoval, G2, G3, val2, val3);
            }

            if (edgetable & 8)
            {
                vertlist[3] = VertexInterp(isoval, P3, P0, val3, val0);
                normlist[3] = GradInterp(isoval, G3, G0, val3, val0);
            }
            
            if (edgetable & 16)
            {
                vertlist[4] = VertexInterp(isoval, P4, P5, val4, val5);
                normlist[4] = GradInterp(isoval, G4, G5, val4, val5);
            }
            
            if (edgetable & 32)
            {
                vertlist[5] = VertexInterp(isoval, P5, P6, val5, val6);
                normlist[5] = GradInterp(isoval, G5, G6, val5, val6);
            }

            if (edgetable & 64)
            {
                vertlist[6] = VertexInterp(isoval, P6, P7, val6, val7);
                normlist[6] = GradInterp(isoval, G6, G7, val6, val7);
            }

            if (edgetable & 128)
            {
                vertlist[7] = VertexInterp(isoval, P7, P4, val7, val4);
                normlist[7] = GradInterp(isoval, G7, G4, val7, val4);
            }
            
            if (edgetable & 256)
            {
                vertlist[8] = VertexInterp(isoval, P0, P4, val0, val4);
                normlist[8] = GradInterp(isoval, G0, G4, val0, val4);
            }

            if (edgetable & 512)
            {
                vertlist[9] = VertexInterp(isoval, P1, P5, val1, val5);
                normlist[9] = GradInterp(isoval, G1, G5, val1, val5);
            }

            if (edgetable & 1024)
            {
                vertlist[10] = VertexInterp(isoval, P2, P6, val2, val6);
                normlist[10] = GradInterp(isoval, G2, G6, val2, val6);
            }

            if (edgetable & 2048)
            {
                vertlist[11] = VertexInterp(isoval, P3, P7, val3, val7);
                normlist[11] = GradInterp(isoval, G3, G7, val3, val7);
            }

            int j;
            int ntri = 0;
            int l;

            __local float3 triangles[30];
            __local float3 normals[30];

            for (j = 0; triTable[16*cubeIndex[gid]+j] != -1; j+=3)
            {
                triangles[3*ntri+0] = vertlist[triTable[16*cubeIndex[gid]+j+0]];
                triangles[3*ntri+1] = vertlist[triTable[16*cubeIndex[gid]+j+1]];
                triangles[3*ntri+2] = vertlist[triTable[16*cubeIndex[gid]+j+2]];

                normals[3*ntri+0] = normlist[triTable[16*cubeIndex[gid]+j+0]];
                normals[3*ntri+1] = normlist[triTable[16*cubeIndex[gid]+j+1]];
                normals[3*ntri+2] = normlist[triTable[16*cubeIndex[gid]+j+2]];

                ntri++;
            }

            triCount[gid] = ntri;
            
            for (l=0; l<ntri; l++)
            {
                // tri[Ntri+l] = triangles[l];
                // norm[Ntri+l] = normals[l];
                // triIdx[3*Ntri+3*l+0] = 3*Ntri+3*l+0;
                // triIdx[3*Ntri+3*l+1] = 3*Ntri+3*l+1;
                // triIdx[3*Ntri+3*l+2] = 3*Ntri+3*l+2;
                tri[15*gid+3*l+0] = triangles[3*l+0];
                tri[15*gid+3*l+1] = triangles[3*l+1];
                tri[15*gid+3*l+2] = triangles[3*l+2];

                norm[15*gid+3*l+0] = normals[3*l+0];
                norm[15*gid+3*l+1] = normals[3*l+1];
                norm[15*gid+3*l+2] = normals[3*l+2];
            }
            
            // Ntri += ntri;
        }
    }
}

// kernel void mc_kernel_polygonisecube(
// 		__global unsigned short* edgetable,
//         __global signed char* tritable,
//         __global unsigned char* cubeindex,
//         __global float* mcgrid,
//         float isolevel,
//         __global float* triangles,
//         __global char* trianglescount)
// {
// 	size_t gid = get_global_id(0);

// 	unsigned char cubeidx = cubeindex[gid];
// 	unsigned short edges = edgetable[cubeidx];

//     float4 vert0 = float4(mcgrid[32*gid+0],  mcgrid[32*gid+1],  mcgrid[32*gid+2],  mcgrid[32*gid+3]);
//     float4 vert1 = float4(mcgrid[32*gid+4],  mcgrid[32*gid+5],  mcgrid[32*gid+6],  mcgrid[32*gid+7]);
//     float4 vert2 = float4(mcgrid[32*gid+8],  mcgrid[32*gid+9],  mcgrid[32*gid+10], mcgrid[32*gid+11]);
//     float4 vert3 = float4(mcgrid[32*gid+12], mcgrid[32*gid+13], mcgrid[32*gid+14], mcgrid[32*gid+15]);
//     float4 vert4 = float4(mcgrid[32*gid+16], mcgrid[32*gid+17], mcgrid[32*gid+18], mcgrid[32*gid+19]);
//     float4 vert5 = float4(mcgrid[32*gid+20], mcgrid[32*gid+21], mcgrid[32*gid+22], mcgrid[32*gid+23]);
//     float4 vert6 = float4(mcgrid[32*gid+24], mcgrid[32*gid+25], mcgrid[32*gid+26], mcgrid[32*gid+27]);
//     float4 vert7 = float4(mcgrid[32*gid+28], mcgrid[32*gid+29], mcgrid[32*gid+30], mcgrid[32*gid+31]);

//     char ntriangle = 0;

//     float3 vertlist[12];

//     // cube is entirely in/out of the surface
//     if (edges == 0)
//     {
//     	ntriangle = 0;
// 	}
//     else
//     {
//     	if (edges & 1)
//         {
//         	vertlist[0] = VertexInterp(isolevel, vert0, vert1);
// 		} else if (edges & 2)
//         {
//         	vertlist[1] = VertexInterp(isolevel, vert1, vert2);
// 		} else if (edges & 4)
//         {
//         	vertlist[2] = VertexInterp(isolevel, vert2, vert3);
// 		} else if (edges & 8)
//         {
//         	vertlist[3] = VertexInterp(isolevel, vert3, vert0);
// 		} else if (edges & 16)
//         {
//         	vertlist[4] = VertexInterp(isolevel, vert4, vert5);
// 		} else if (edges & 32)
//         {
//         	vertlist[5] = VertexInterp(isolevel, vert5, vert6);
// 		} else if (edges & 64)
//         {
//         	vertlist[6] = VertexInterp(isolevel, vert6, vert7);
// 		} else if (edges & 128)
//         {
//         	vertlist[7] = VertexInterp(isolevel, vert7, vert4);
// 		} else if (edges & 256)
//         {
//         	vertlist[8] = VertexInterp(isolevel, vert0, vert4);
// 		} else if (edges & 512)
//         {
//         	vertlist[9] = VertexInterp(isolevel, vert1, vert5);
// 		} else if (edges & 1024)
//         {
//         	vertlist[10] = VertexInterp(isolevel, vert2, vert6);
// 		} else if (edges & 2048)
//         {
//         	vertlist[11] = VertexInterp(isolevel, vert3, vert7);
// 		}
            
//         int i;
            
//         for (i=0; tritable[cubeidx*16+i] != -1; i+=3)
//         {
//         	triangles[45*gid+9*i+0] = vertlist[tritable[cubeidx * 16 + i]].x;
//             triangles[45*gid+9*i+1] = vertlist[tritable[cubeidx * 16 + i]].y;
//             triangles[45*gid+9*i+2] = vertlist[tritable[cubeidx * 16 + i]].z;
                
//             triangles[45*gid+9*i+3] = vertlist[tritable[cubeidx * 16 + i + 1]].x;
//             triangles[45*gid+9*i+4] = vertlist[tritable[cubeidx * 16 + i + 1]].y;
//             triangles[45*gid+9*i+5] = vertlist[tritable[cubeidx * 16 + i + 1]].z;

//             triangles[45*gid+9*i+6] = vertlist[tritable[cubeidx * 16 + i + 2]].x;
//             triangles[45*gid+9*i+7] = vertlist[tritable[cubeidx * 16 + i + 2]].y;
//             triangles[45*gid+9*i+8] = vertlist[tritable[cubeidx * 16 + i + 2]].z;

//             ntriangle++;
// 		}
// 	}
        
//     trianglescount[gid] = ntriangle;  
// }

// kernel void mc_kernel_packtriangles(
// 		__global float* triangles,
//         __global char* trianglescount,
//         __global float* trianglesRender)
// {
// 	size_t gid = get_global_id(0);

// 	trianglesRender[45*gid+0] = triangles[45*gid+0];
// 	trianglesRender[45*gid+1] = triangles[45*gid+1];
//     trianglesRender[45*gid+2] = triangles[45*gid+2];
// }

kernel void sph_kernel_calcRelaxPos(
		float m,
        float cellSize,
        float dt,
        float dt2,
        float kNearNorm,
        float kNorm,
        float kSurfaceTension,
        float kLinearViscocity,
        float kQuadraticViscocity,
        int maxNeighbors,
        __global float4* pos,
        __global float4* vel,
        __global int* neighborMap,
        __global float* density,
        __global float* nearDensity,
        __global float* pressure,
        __global float* nearPressure,
        __global float4* relaxPos)
{
	size_t gid = get_global_id(0);

	float3 myPos = pos[gid].xyz;
    float3 myVel = vel[gid].xyz;

    float x = myPos.x;
    float y = myPos.y;
    float z = myPos.z;

    int i;
    float3 posJ;
    float3 velJ;

    for (i = 0; i < maxNeighbors; i++)
    {
    	int nID = neighborMap[maxNeighbors * gid+i];
        if ((nID >= 0) && ((unsigned int)nID == gid))
        	continue;

		posJ = pos[nID].xyz;

        float3 diff = posJ - myPos;
        float r = length(diff);

        float dx = diff.x;
        float dy = diff.y;
        float dz = diff.z;

        float a = 1 - r/cellSize;

        float d = dt2 * ((nearPressure[gid] + nearPressure[nID]) * a * a * a * kNearNorm + (pressure[gid] + pressure[nID]) * a * a * kNorm) / 2.0;

        x -= d * dx / (r * m);
        y -= d * dy / (r * m);
        z -= d * dz / (r * m);

        x += (kSurfaceTension/m) * m * a * a * kNorm * dx;
        y += (kSurfaceTension/m) * m * a * a * kNorm * dy;
        z += (kSurfaceTension/m) * m * a * a * kNorm * dz;

        velJ = vel[nID].xyz;

        float3 diffV = myVel - velJ;
        float u = diffV.x * dx + diffV.y * dy + diffV.z * dz;

        if (u > 0)
        {
        	u /= r;

            float a = 1 - r/cellSize;

            float I = .5 * dt * a * (kLinearViscocity * u + kQuadraticViscocity * u * u);

            x -= I * dx * dt;
            y -= I * dy * dt;
            z -= I * dz * dt;
		}
	}

    relaxPos[gid].x = x;
    relaxPos[gid].y = y;
    relaxPos[gid].z = z;	
}

kernel void sph_kernel_moveToRelaxPos(
		float dt,
        __global float4* pos,
        __global float4* prevPos,
        __global float4* vel,
        __global float4* relaxPos)
{
	size_t gid = get_global_id(0);

	pos[gid].x = relaxPos[gid].x;
	pos[gid].y = relaxPos[gid].y;
    pos[gid].z = relaxPos[gid].z;

    vel[gid].x = (pos[gid].x - prevPos[gid].x) / dt;
    vel[gid].y = (pos[gid].y - prevPos[gid].y) / dt;
    vel[gid].z = (pos[gid].z - prevPos[gid].z) / dt;
}

kernel void sph_kernel_resolveCollisions(
		float dt,
        float pRadius,
        float cellSize,
        float viewW,
        float viewH,
        float viewD,
        __global float4* pos,
        __global float4* vel)
{
	size_t gid = get_global_id(0);

	float3 myPos = pos[gid].xyz;
    float3 myVel = vel[gid].xyz;

    float3 center         = float3(0.0);
    float3 boxSize        = float3(viewW/2-cellSize-pRadius, viewH/2-cellSize-pRadius, viewD/2-cellSize-pRadius);
        
    float3 xLocal = myPos - center;
    float3 contactPointLocal = min(boxSize, max(-boxSize, xLocal));
    float3 contactPoint = contactPointLocal + center;
    float distance = length(contactPoint - myPos);
        
    if (distance > 0.0 && length(myVel) > 0.0) 
    {
    	float3 normal = normalize(sign(contactPointLocal - xLocal));
        float restitution = .5*distance / (dt * length(myVel));
            
        vel[gid].xyz -= (float3)((1.0 + restitution) * dot(myVel, normal)) * normal;
        pos[gid].xyz = contactPoint;
	}
}
