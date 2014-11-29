#version 410 core

#extension GL_ARB_tessellation_shader : enable

// number of CPs in patch
// layout (vertices = 3) out;
layout (vertices = 1) out;

// from VS (use empty modifier [] so we can say anything)
in vec3 position_eye_[];
in vec3 normal_eye_[];
// in vec3 vp[];

struct OutputPatch
{
    vec3 WorldPos_B030;
    vec3 WorldPos_B021;
    vec3 WorldPos_B012;
    vec3 WorldPos_B003;
    vec3 WorldPos_B102;
    vec3 WorldPos_B201;
    vec3 WorldPos_B300;
    vec3 WorldPos_B210;
    vec3 WorldPos_B120;
    vec3 WorldPos_B111;
    vec3 Normal[3];
};

out vec3 _position_eye[], _normal_eye[];//, _vp[];
patch out OutputPatch oPatch;

uniform float tess_fac_inner = 1.0; // controlled by keyboard buttons
uniform float tess_fac_outer = 1.0; // controlled by keyboard buttons

vec3 ProjectToPlane(vec3 Point, vec3 PlanePoint, vec3 PlaneNormal)
{
    vec3 v = Point - PlanePoint;
    float Len = dot(v, PlaneNormal);
    vec3 d = Len * PlaneNormal;
    return (Point - d);
}

void CalcPositions()
{
    // The original vertices stay the same
    oPatch.WorldPos_B030 = position_eye_[0];
    oPatch.WorldPos_B003 = position_eye_[1];
    oPatch.WorldPos_B300 = position_eye_[2];

    // Edges are names according to the opposing vertex
    vec3 EdgeB300 = oPatch.WorldPos_B003 - oPatch.WorldPos_B030;
    vec3 EdgeB030 = oPatch.WorldPos_B300 - oPatch.WorldPos_B003;
    vec3 EdgeB003 = oPatch.WorldPos_B030 - oPatch.WorldPos_B300;

    // Generate two midpoints on each edge
    oPatch.WorldPos_B021 = oPatch.WorldPos_B030 + EdgeB300 / 3.0;
    oPatch.WorldPos_B012 = oPatch.WorldPos_B030 + EdgeB300 * 2.0 / 3.0;
    oPatch.WorldPos_B102 = oPatch.WorldPos_B003 + EdgeB030 / 3.0;
    oPatch.WorldPos_B201 = oPatch.WorldPos_B003 + EdgeB030 * 2.0 / 3.0;
    oPatch.WorldPos_B210 = oPatch.WorldPos_B300 + EdgeB003 / 3.0;
    oPatch.WorldPos_B120 = oPatch.WorldPos_B300 + EdgeB003 * 2.0 / 3.0;

    // Project each midpoint on the plane defined by the nearest vertex and its normal
    oPatch.WorldPos_B021 = ProjectToPlane(oPatch.WorldPos_B021, oPatch.WorldPos_B030,
                                          oPatch.Normal[0]);
    oPatch.WorldPos_B012 = ProjectToPlane(oPatch.WorldPos_B012, oPatch.WorldPos_B003,
                                         oPatch.Normal[1]);
    oPatch.WorldPos_B102 = ProjectToPlane(oPatch.WorldPos_B102, oPatch.WorldPos_B003,
                                         oPatch.Normal[1]);
    oPatch.WorldPos_B201 = ProjectToPlane(oPatch.WorldPos_B201, oPatch.WorldPos_B300,
                                         oPatch.Normal[2]);
    oPatch.WorldPos_B210 = ProjectToPlane(oPatch.WorldPos_B210, oPatch.WorldPos_B300,
                                         oPatch.Normal[2]);
    oPatch.WorldPos_B120 = ProjectToPlane(oPatch.WorldPos_B120, oPatch.WorldPos_B030,
                                         oPatch.Normal[0]);

    // Handle the center
    vec3 Center = (oPatch.WorldPos_B003 + oPatch.WorldPos_B030 + oPatch.WorldPos_B300) / 3.0;
    oPatch.WorldPos_B111 = (oPatch.WorldPos_B021 + oPatch.WorldPos_B012 + oPatch.WorldPos_B102 +
                          oPatch.WorldPos_B201 + oPatch.WorldPos_B210 + oPatch.WorldPos_B120) / 6.0;
    oPatch.WorldPos_B111 += (oPatch.WorldPos_B111 - Center) / 2.0;
}

void main () {
	_position_eye[gl_InvocationID] = position_eye_[gl_InvocationID];
	_normal_eye[gl_InvocationID] = normal_eye_[gl_InvocationID];
	// _vp[gl_InvocationID] = vp[gl_InvocationID];

	// Set the control points of the output patch
    for (int i = 0 ; i < 3 ; i++) {
       oPatch.Normal[i] = normal_eye_[i];
    }

    CalcPositions();

    // Calculate the tessellation levels
	gl_TessLevelInner[0] = tess_fac_inner; // number of nested primitives to generate
	gl_TessLevelOuter[0] = tess_fac_outer; // times to subdivide first side
	gl_TessLevelOuter[1] = tess_fac_outer; // times to subdivide second side
	gl_TessLevelOuter[2] = tess_fac_outer; // times to subdivide third side

}
